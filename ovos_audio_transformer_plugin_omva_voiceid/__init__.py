"""
OMVA Voice Identification Plugin for OVOS

This plugin provides multi-user voice identification capabilities for the OVOS ecosystem,
enabling automatic speaker recognition and user context switching.

The plugin extends OVOS AudioTransformer to identify speakers in real-time audio streams
and emit user identification events to the message bus for session management.
"""

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from ovos_bus_client import MessageBusClient, Message
from ovos_plugin_manager.templates.transformers import AudioTransformer
from ovos_utils.log import LOG
from ovos_utils.xdg_utils import xdg_data_home
from speech_recognition import AudioData

from .version import VERSION_ALPHA, VERSION_BUILD, VERSION_MAJOR, VERSION_MINOR
from .voice_processor import OMVAVoiceProcessor

__version__ = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_BUILD}"
if VERSION_ALPHA > 0:
    __version__ += f"a{VERSION_ALPHA}"


class OMVAVoiceIDPlugin(AudioTransformer):
    """
    OMVA Voice Identification AudioTransformer Plugin

    This plugin identifies speakers in audio streams and emits user identification
    events to the OVOS message bus. It integrates seamlessly with the OVOS audio
    pipeline without modifying the core audio data flow.

    Features:
    - Real-time speaker identification using SpeechBrain ECAPA-TDNN model
    - Configurable confidence thresholds and fallback mechanisms
    - Multi-user session management integration
    - Voice enrollment and model training support
    - Performance optimized for real-time processing
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the OMVA Voice Identification Plugin

        Args:
            config: Plugin configuration dictionary
        """
        config = config or {}

        # Initialize with priority 10 (standard for audio transformers)
        super().__init__("ovos-audio-transformer-plugin-omva-voiceid", 10, config)

        # Plugin configuration
        self.confidence_threshold = self.config.get("confidence_threshold", 0.8)
        self.enable_enrollment = self.config.get("enable_enrollment", True)
        self.cache_dir = (
            self.config.get("cache_dir") or f"{xdg_data_home()}/omva_voiceid"
        )
        self.processing_timeout_ms = self.config.get("processing_timeout_ms", 100)

        # Voice processing configuration
        voice_config = self.config.get("voice_processing", {})
        self.mfcc_coefficients = voice_config.get("mfcc_coefficients", 13)
        self.window_size = voice_config.get("window_size", 0.025)
        self.hop_length = voice_config.get("hop_length", 0.01)
        self.sample_rate = voice_config.get("sample_rate", 16000)

        # Model configuration
        self.model_source = config.get("model", "speechbrain/spkrec-ecapa-voxceleb")
        self.model_cache_dir = config.get(
            "model_cache_dir", "./models/speechbrain_cache"
        )
        self.verification_threshold = config.get("verification_threshold", 0.25)
        self.sample_rate = config.get("sample_rate", 16000)
        self.gpu = config.get("gpu", False)

        # Performance tracking
        self.processing_stats = {
            "total_processed": 0,
            "successful_identifications": 0,
            "failed_identifications": 0,
            "average_processing_time_ms": 0.0,
        }

        # Initialize voice processor
        self.voice_processor = None
        self._initialize_voice_processor()

        LOG.info(f"OMVA Voice ID Plugin v{__version__} initialized")
        LOG.info(
            f"Configuration: threshold={self.confidence_threshold}, "
            f"cache_dir={self.cache_dir}"
        )

    def _initialize_voice_processor(self):
        """Initialize the OMVA voice processing components"""
        try:
            processor_config = {
                "mfcc_coefficients": self.mfcc_coefficients,
                "window_size": self.window_size,
                "hop_length": self.hop_length,
                "sample_rate": self.sample_rate,
                "cache_dir": self.cache_dir,
                "model_source": self.model_source,
                "model_cache_dir": self.model_cache_dir,
                "verification_threshold": self.verification_threshold,
                "use_gpu": self.gpu,
            }

            self.voice_processor = OMVAVoiceProcessor(processor_config)
            LOG.info("OMVA voice processor initialized successfully")

        except Exception as e:
            LOG.error(f"Failed to initialize OMVA voice processor: {e}")
            # Continue without voice processing capability
            self.voice_processor = None

    @staticmethod
    def audiochunk2array(audio_data: bytes) -> torch.Tensor:
        """
        Convert audio chunk to PyTorch tensor format for SpeechBrain

        Args:
            audio_data: Raw audio bytes (16-bit PCM)

        Returns:
            Normalized float32 PyTorch tensor
        """
        # Convert buffer to int16 array
        audio_as_np_int16 = np.frombuffer(audio_data, dtype=np.int16)

        # Convert to float32 and normalize to [-1.0, 1.0] range
        audio_as_np_float32 = audio_as_np_int16.astype(np.float32)
        max_int16 = 2**15
        normalized_audio = audio_as_np_float32 / max_int16

        # Convert to PyTorch tensor
        return torch.from_numpy(normalized_audio).float()

    def identify_speaker(self, audio_data: bytes) -> Tuple[Optional[str], float]:
        """
        Identify speaker from audio data using SpeechBrain

        Args:
            audio_data: Raw audio bytes

        Returns:
            Tuple of (speaker_id, confidence) or (None, 0.0) if identification fails
        """
        if self.voice_processor is None:
            LOG.warning("Voice processor not initialized, skipping identification")
            return None, 0.0

        try:
            start_time = time.time()

            # Convert audio to tensor format
            audio_tensor = self.audiochunk2array(audio_data)

            # Check for minimum audio length (avoid processing too short clips)
            min_audio_length = int(0.5 * self.sample_rate)  # 0.5 seconds minimum
            if audio_tensor.size(0) < min_audio_length:
                return None, 0.0

            # Run voice identification using SpeechBrain
            speaker_id, confidence = self.voice_processor.identify_speaker(audio_tensor)

            # Update performance stats
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_processing_stats(processing_time_ms, speaker_id is not None)

            # Check processing timeout
            if processing_time_ms > self.processing_timeout_ms:
                LOG.warning(
                    f"Voice identification took {processing_time_ms:.1f}ms "
                    f"(threshold: {self.processing_timeout_ms}ms)"
                )

            return speaker_id, confidence

        except Exception as e:
            LOG.error(f"Speaker identification failed: {e}")
            self._update_processing_stats(0, False)
            return None, 0.0

    def _update_processing_stats(self, processing_time_ms: float, success: bool):
        """Update internal processing statistics"""
        self.processing_stats["total_processed"] += 1

        if success:
            self.processing_stats["successful_identifications"] += 1
        else:
            self.processing_stats["failed_identifications"] += 1

        # Update running average of processing time
        current_avg = self.processing_stats["average_processing_time_ms"]
        total_processed = self.processing_stats["total_processed"]

        new_avg = (
            (current_avg * (total_processed - 1)) + processing_time_ms
        ) / total_processed
        self.processing_stats["average_processing_time_ms"] = new_avg

    def _emit_identification_event(self, speaker_id: Optional[str], confidence: float):
        """
        Emit voice identification event to message bus

        Args:
            speaker_id: Identified speaker ID or None
            confidence: Identification confidence score
        """
        if not self.bus:
            return

        if speaker_id and confidence >= self.confidence_threshold:
            # Successful identification
            event_data = {
                "speaker_id": speaker_id,
                "confidence": confidence,
                "processing_time_ms": self.processing_stats[
                    "average_processing_time_ms"
                ],
                "plugin_version": __version__,
                "timestamp": time.time(),
            }

            self.bus.emit(Message("ovos.voice.identified", event_data))
            LOG.info(f"Speaker identified: {speaker_id} (confidence: {confidence:.3f})")

        else:
            # Failed or low-confidence identification
            event_data = {
                "confidence": confidence,
                "speaker_candidates": [],  # Could be populated with top candidates
                "fallback_mode": "guest",
                "plugin_version": __version__,
                "timestamp": time.time(),
            }

            if speaker_id:
                # Low confidence but had a candidate
                event_data["speaker_candidates"] = [
                    {"speaker_id": speaker_id, "confidence": confidence}
                ]

            self.bus.emit(Message("ovos.voice.unknown", event_data))
            LOG.debug(
                f"Speaker identification failed or low confidence: {confidence:.3f}"
            )

    def bind(self, bus=None):
        """
        Bind to message bus and register event handlers

        Args:
            bus: OVOS message bus instance
        """
        super().bind(bus)

        if self.bus:
            # Register message handlers for plugin management
            self.bus.on("ovos.voiceid.get_stats", self.handle_get_stats)
            self.bus.on("ovos.voiceid.reset_stats", self.handle_reset_stats)

            if self.enable_enrollment:
                self.bus.on("ovos.voiceid.enroll_user", self.handle_enroll_user)
                self.bus.on("ovos.voiceid.list_users", self.handle_list_users)

            LOG.info("OMVA Voice ID plugin bound to message bus")

    def handle_get_stats(self, message: Message):
        """Handle request for plugin statistics"""
        stats_data = {
            **self.processing_stats,
            "plugin_version": __version__,
            "configuration": {
                "confidence_threshold": self.confidence_threshold,
                "processing_timeout_ms": self.processing_timeout_ms,
                "sample_rate": self.sample_rate,
            },
        }

        self.bus.emit(Message("ovos.voiceid.stats.response", stats_data))

    def handle_reset_stats(self, message: Message):
        """Handle request to reset statistics"""
        self.processing_stats = {
            "total_processed": 0,
            "successful_identifications": 0,
            "failed_identifications": 0,
            "average_processing_time_ms": 0.0,
        }

        self.bus.emit(Message("ovos.voiceid.stats.reset", {"status": "success"}))
        LOG.info("Plugin statistics reset")

    def handle_enroll_user(self, message: Message):
        """Handle user enrollment request"""
        # Placeholder for user enrollment functionality
        user_id = message.data.get("user_id")
        audio_samples = message.data.get("audio_samples", [])

        LOG.info(f"User enrollment request received for: {user_id}")
        # TODO: Implement enrollment logic

        self.bus.emit(
            Message(
                "ovos.voiceid.enroll.response",
                {
                    "user_id": user_id,
                    "status": "not_implemented",
                    "message": "User enrollment not yet implemented",
                },
            )
        )

    def handle_list_users(self, message: Message):
        """Handle request to list enrolled users"""
        # Placeholder for user listing functionality
        LOG.info("User list request received")
        # TODO: Implement user listing logic

        self.bus.emit(
            Message(
                "ovos.voiceid.users.response",
                {
                    "users": [],
                    "status": "not_implemented",
                    "message": "User listing not yet implemented",
                },
            )
        )

    def transform(self, audio_data) -> Any:
        """
        Main AudioTransformer method - processes audio and emits identification events

        This method is called by the OVOS audio pipeline for each audio chunk.
        It performs voice identification and emits events to the message bus while
        passing the original audio data through unchanged.

        Args:
            audio_data: Audio data (bytes or AudioData object)

        Returns:
            Original audio data unchanged (pass-through)
        """
        # Handle both raw bytes and AudioData objects
        if isinstance(audio_data, AudioData):
            raw_audio = audio_data.get_wav_data()
        else:
            raw_audio = audio_data

        # Skip processing if audio is too small
        if len(raw_audio) < 1024:  # Minimum viable audio chunk
            return audio_data

        # Perform voice identification
        speaker_id, confidence = self.identify_speaker(raw_audio)

        # Emit identification event to message bus
        self._emit_identification_event(speaker_id, confidence)

        # Return original audio data unchanged (pass-through behavior)
        return audio_data

    def shutdown(self):
        """Cleanup resources on plugin shutdown"""
        LOG.info("Shutting down OMVA Voice ID plugin")

        if self.voice_processor:
            self.voice_processor.cleanup()

        # Log final statistics
        total = self.processing_stats["total_processed"]
        successful = self.processing_stats["successful_identifications"]
        avg_time = self.processing_stats["average_processing_time_ms"]

        LOG.info(
            f"Plugin shutdown - Processed: {total}, Successful: {successful}, "
            f"Avg time: {avg_time:.1f}ms"
        )

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.shutdown()
        except:
            pass  # Ignore cleanup errors during destruction
