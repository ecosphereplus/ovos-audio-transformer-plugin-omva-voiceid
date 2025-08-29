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
from ovos_bus_client import Message
from ovos_plugin_manager.templates.transformers import AudioTransformer
from ovos_utils.log import LOG
from ovos_utils.xdg_utils import xdg_data_home
from speech_recognition import AudioData

from ovos_audio_transformer_plugin_omva_voiceid.version import (
    VERSION_ALPHA,
    VERSION_BUILD,
    VERSION_MAJOR,
    VERSION_MINOR,
)
from ovos_audio_transformer_plugin_omva_voiceid.voice_processor import (
    OMVAVoiceProcessor,
)

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
        self.processing_timeout_ms = self.config.get("processing_timeout_ms", 100)

        # Model configuration
        self.model_source = config.get("model", "speechbrain/spkrec-ecapa-voxceleb")
        self.model_cache_dir = config.get(
            "model_cache_dir", f"{xdg_data_home()}/omva_voiceid"
        )
        self.confidence_threshold = config.get("confidence_threshold", 0.8)
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
            f"model_cache_dir={self.model_cache_dir}"
        )

    def _initialize_voice_processor(self):
        """Initialize the OMVA voice processing components"""
        try:
            processor_config = {
                "sample_rate": self.sample_rate,
                "model_source": self.model_source,
                "model_cache_dir": self.model_cache_dir,
                "confidence_threshold": self.confidence_threshold,
                "gpu": self.gpu,
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

        if speaker_id:
            # Successful identification (voice processor already checked threshold)
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
            # Failed identification
            event_data = {
                "confidence": confidence,
                "speaker_candidates": [],  # Could be populated with top candidates
                "fallback_mode": "guest",
                "plugin_version": __version__,
                "timestamp": time.time(),
            }

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
                self.bus.on("ovos.voiceid.remove_user", self.handle_remove_user)
                self.bus.on("ovos.voiceid.update_user", self.handle_update_user)
                self.bus.on("ovos.voiceid.get_user_info", self.handle_get_user_info)
                self.bus.on("ovos.voiceid.verify_speakers", self.handle_verify_speakers)

            LOG.info("OMVA Voice ID plugin bound to message bus")

    def handle_get_stats(self, _: Message):
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

        if self.bus:
            self.bus.emit(Message("ovos.voiceid.stats.response", stats_data))

    def handle_reset_stats(self, _: Message):
        """Handle request to reset statistics"""
        self.processing_stats = {
            "total_processed": 0,
            "successful_identifications": 0,
            "failed_identifications": 0,
            "average_processing_time_ms": 0.0,
        }
        if self.bus:
            self.bus.emit(Message("ovos.voiceid.stats.reset", {"status": "success"}))
        LOG.info("Plugin statistics reset")

    def handle_enroll_user(self, message: Message):
        """Handle user enrollment request"""
        user_id = message.data.get("user_id")
        audio_samples = message.data.get("audio_samples", [])

        LOG.info(f"User enrollment request received for: {user_id}")

        if not user_id:
            if self.bus:
                self.bus.emit(
                    Message(
                        "ovos.voiceid.enroll.response",
                        {
                            "user_id": None,
                            "status": "error",
                            "message": "User ID is required for enrollment",
                        },
                    )
                )
            return

        if not audio_samples:
            if self.bus:
                self.bus.emit(
                    Message(
                        "ovos.voiceid.enroll.response",
                        {
                            "user_id": user_id,
                            "status": "error",
                            "message": "Audio samples are required for enrollment",
                        },
                    )
                )
            return

        if self.voice_processor is None:
            if self.bus:
                self.bus.emit(
                    Message(
                        "ovos.voiceid.enroll.response",
                        {
                            "user_id": user_id,
                            "status": "error",
                            "message": "Voice processor not initialized",
                        },
                    )
                )
            return

        try:
            # Convert audio samples from hex strings to tensors
            audio_tensors = []
            for sample_hex in audio_samples:
                if isinstance(sample_hex, str):
                    # Convert hex string back to bytes, then to audio tensor
                    audio_bytes = bytes.fromhex(sample_hex)
                    audio_tensor = self.audiochunk2array(audio_bytes)
                    audio_tensors.append(audio_tensor)
                else:
                    # Assume it's already audio data
                    if isinstance(sample_hex, bytes):
                        audio_tensor = self.audiochunk2array(sample_hex)
                        audio_tensors.append(audio_tensor)

            if not audio_tensors:
                raise ValueError("No valid audio tensors could be created")

            # Use voice processor to enroll user
            success = self.voice_processor.enroll_user(user_id, audio_tensors)

            if success:
                LOG.info(f"User {user_id} enrolled successfully")
                if self.bus:
                    self.bus.emit(
                        Message(
                            "ovos.voiceid.enroll.response",
                            {
                                "user_id": user_id,
                                "status": "success",
                                "message": f"User {user_id} enrolled successfully with {len(audio_tensors)} audio samples",
                                "samples_processed": len(audio_tensors),
                            },
                        )
                    )
            else:
                LOG.error(f"Failed to enroll user {user_id}")
                if self.bus:
                    self.bus.emit(
                        Message(
                            "ovos.voiceid.enroll.response",
                            {
                                "user_id": user_id,
                                "status": "error",
                                "message": f"Enrollment failed for user {user_id}",
                            },
                        )
                    )

        except Exception as e:
            LOG.error(f"Enrollment error for user {user_id}: {e}")
            if self.bus:
                self.bus.emit(
                    Message(
                        "ovos.voiceid.enroll.response",
                        {
                            "user_id": user_id,
                            "status": "error",
                            "message": f"Enrollment failed: {str(e)}",
                        },
                    )
                )

    def handle_list_users(self, _: Message):
        """Handle request to list enrolled users"""
        LOG.info("User list request received")

        if self.voice_processor is None:
            if self.bus:
                self.bus.emit(
                    Message(
                        "ovos.voiceid.users.response",
                        {
                            "users": [],
                            "status": "error",
                            "message": "Voice processor not initialized",
                        },
                    )
                )
            return

        try:
            # Get enrolled users from voice processor
            enrolled_users = self.voice_processor.get_enrolled_users()

            # Get additional model information
            model_info = self.voice_processor.get_model_info()

            LOG.info(f"Retrieved {len(enrolled_users)} enrolled users")

            if self.bus:
                self.bus.emit(
                    Message(
                        "ovos.voiceid.users.response",
                        {
                            "users": enrolled_users,
                            "status": "success",
                            "total_users": len(enrolled_users),
                            "model_info": {
                                "model_source": model_info.get("model_source"),
                                "model_available": model_info.get(
                                    "model_available", False
                                ),
                                "confidence_threshold": model_info.get(
                                    "confidence_threshold"
                                ),
                                "sample_rate": model_info.get("sample_rate"),
                            },
                            "message": f"Retrieved {len(enrolled_users)} enrolled users",
                        },
                    )
                )

        except Exception as e:
            LOG.error(f"Failed to retrieve user list: {e}")
            if self.bus:
                self.bus.emit(
                    Message(
                        "ovos.voiceid.users.response",
                        {
                            "users": [],
                            "status": "error",
                            "message": f"Failed to retrieve users: {str(e)}",
                        },
                    )
                )

    def handle_remove_user(self, message: Message):
        """Handle user removal request"""
        user_id = message.data.get("user_id")

        LOG.info(f"User removal request received for: {user_id}")

        if not user_id:
            if self.bus:
                self.bus.emit(
                    Message(
                        "ovos.voiceid.remove.response",
                        {
                            "user_id": None,
                            "status": "error",
                            "message": "User ID is required for removal",
                        },
                    )
                )
            return

        if self.voice_processor is None:
            if self.bus:
                self.bus.emit(
                    Message(
                        "ovos.voiceid.remove.response",
                        {
                            "user_id": user_id,
                            "status": "error",
                            "message": "Voice processor not initialized",
                        },
                    )
                )
            return

        try:
            # Use voice processor to remove user
            success = self.voice_processor.remove_user(user_id)

            if success:
                LOG.info(f"User {user_id} removed successfully")
                if self.bus:
                    self.bus.emit(
                        Message(
                            "ovos.voiceid.remove.response",
                            {
                                "user_id": user_id,
                                "status": "success",
                                "message": f"User {user_id} removed successfully",
                            },
                        )
                    )
            else:
                LOG.warning(f"User {user_id} not found for removal")
                if self.bus:
                    self.bus.emit(
                        Message(
                            "ovos.voiceid.remove.response",
                            {
                                "user_id": user_id,
                                "status": "not_found",
                                "message": f"User {user_id} not found in enrolled users",
                            },
                        )
                    )

        except Exception as e:
            LOG.error(f"User removal error for {user_id}: {e}")
            if self.bus:
                self.bus.emit(
                    Message(
                        "ovos.voiceid.remove.response",
                        {
                            "user_id": user_id,
                            "status": "error",
                            "message": f"Removal failed: {str(e)}",
                        },
                    )
                )

    def handle_verify_speakers(self, message: Message):
        """Handle speaker verification request (comparing two audio samples)"""
        audio_sample1 = message.data.get("audio_sample1")
        audio_sample2 = message.data.get("audio_sample2")

        LOG.info("Speaker verification request received")

        if not audio_sample1 or not audio_sample2:
            if self.bus:
                self.bus.emit(
                    Message(
                        "ovos.voiceid.verify.response",
                        {
                            "is_same_speaker": False,
                            "similarity_score": 0.0,
                            "status": "error",
                            "message": "Two audio samples are required for verification",
                        },
                    )
                )
            return

        if self.voice_processor is None:
            if self.bus:
                self.bus.emit(
                    Message(
                        "ovos.voiceid.verify.response",
                        {
                            "is_same_speaker": False,
                            "similarity_score": 0.0,
                            "status": "error",
                            "message": "Voice processor not initialized",
                        },
                    )
                )
            return

        try:
            # Convert audio samples from hex strings to tensors
            if isinstance(audio_sample1, str):
                audio_bytes1 = bytes.fromhex(audio_sample1)
                audio_tensor1 = self.audiochunk2array(audio_bytes1)
            else:
                audio_tensor1 = self.audiochunk2array(audio_sample1)

            if isinstance(audio_sample2, str):
                audio_bytes2 = bytes.fromhex(audio_sample2)
                audio_tensor2 = self.audiochunk2array(audio_bytes2)
            else:
                audio_tensor2 = self.audiochunk2array(audio_sample2)

            # Use voice processor to verify speakers
            is_same_speaker, similarity_score = self.voice_processor.verify_speakers(
                audio_tensor1, audio_tensor2
            )

            LOG.info(
                f"Speaker verification result: same_speaker={is_same_speaker}, score={similarity_score:.3f}"
            )

            if self.bus:
                self.bus.emit(
                    Message(
                        "ovos.voiceid.verify.response",
                        {
                            "is_same_speaker": is_same_speaker,
                            "similarity_score": similarity_score,
                            "status": "success",
                            "confidence_threshold": self.confidence_threshold,
                            "message": f"Verification complete: {'Same speaker' if is_same_speaker else 'Different speakers'} (score: {similarity_score:.3f})",
                        },
                    )
                )

        except Exception as e:
            LOG.error(f"Speaker verification error: {e}")
            if self.bus:
                self.bus.emit(
                    Message(
                        "ovos.voiceid.verify.response",
                        {
                            "is_same_speaker": False,
                            "similarity_score": 0.0,
                            "status": "error",
                            "message": f"Verification failed: {str(e)}",
                        },
                    )
                )

    def handle_update_user(self, message: Message):
        """Handle user profile update request (re-enrollment with new samples)"""
        user_id = message.data.get("user_id")
        audio_samples = message.data.get("audio_samples", [])
        update_mode = message.data.get("mode", "replace")  # "replace" or "append"

        LOG.info(f"User update request received for: {user_id} (mode: {update_mode})")

        if not user_id:
            if self.bus:
                self.bus.emit(
                    Message(
                        "ovos.voiceid.update.response",
                        {
                            "user_id": None,
                            "status": "error",
                            "message": "User ID is required for update",
                        },
                    )
                )
            return

        if not audio_samples:
            if self.bus:
                self.bus.emit(
                    Message(
                        "ovos.voiceid.update.response",
                        {
                            "user_id": user_id,
                            "status": "error",
                            "message": "Audio samples are required for update",
                        },
                    )
                )
            return

        if self.voice_processor is None:
            if self.bus:
                self.bus.emit(
                    Message(
                        "ovos.voiceid.update.response",
                        {
                            "user_id": user_id,
                            "status": "error",
                            "message": "Voice processor not initialized",
                        },
                    )
                )
            return

        try:
            # Check if user exists
            enrolled_users = self.voice_processor.get_enrolled_users()
            if user_id not in enrolled_users:
                if self.bus:
                    self.bus.emit(
                        Message(
                            "ovos.voiceid.update.response",
                            {
                                "user_id": user_id,
                                "status": "not_found",
                                "message": f"User {user_id} not found. Use enrollment instead.",
                                "suggestion": "Use ovos.voiceid.enroll_user for new users",
                            },
                        )
                    )
                return

            # Convert audio samples from hex strings to tensors
            audio_tensors = []
            for sample_hex in audio_samples:
                if isinstance(sample_hex, str):
                    # Convert hex string back to bytes, then to audio tensor
                    audio_bytes = bytes.fromhex(sample_hex)
                    audio_tensor = self.audiochunk2array(audio_bytes)
                    audio_tensors.append(audio_tensor)
                else:
                    # Assume it's already audio data
                    if isinstance(sample_hex, bytes):
                        audio_tensor = self.audiochunk2array(sample_hex)
                        audio_tensors.append(audio_tensor)

            if not audio_tensors:
                raise ValueError("No valid audio tensors could be created")

            # For update, we use re-enrollment with new samples
            # The voice processor will replace the existing embedding
            LOG.info(
                f"Updating user {user_id} with {len(audio_tensors)} new audio samples"
            )

            # Use voice processor to re-enroll user (this updates the existing profile)
            success = self.voice_processor.enroll_user(user_id, audio_tensors)

            if success:
                LOG.info(f"User {user_id} updated successfully")
                if self.bus:
                    self.bus.emit(
                        Message(
                            "ovos.voiceid.update.response",
                            {
                                "user_id": user_id,
                                "status": "success",
                                "message": f"User {user_id} profile updated successfully with {len(audio_tensors)} new audio samples",
                                "samples_processed": len(audio_tensors),
                                "mode": update_mode,
                                "previous_profile": "replaced",
                            },
                        )
                    )
            else:
                LOG.error(f"Failed to update user {user_id}")
                if self.bus:
                    self.bus.emit(
                        Message(
                            "ovos.voiceid.update.response",
                            {
                                "user_id": user_id,
                                "status": "error",
                                "message": f"Profile update failed for user {user_id}",
                            },
                        )
                    )

        except Exception as e:
            LOG.error(f"User update error for {user_id}: {e}")
            if self.bus:
                self.bus.emit(
                    Message(
                        "ovos.voiceid.update.response",
                        {
                            "user_id": user_id,
                            "status": "error",
                            "message": f"Update failed: {str(e)}",
                        },
                    )
                )

    def handle_get_user_info(self, message: Message):
        """Handle request for detailed user information"""
        user_id = message.data.get("user_id")

        LOG.info(f"User info request received for: {user_id}")

        if not user_id:
            if self.bus:
                self.bus.emit(
                    Message(
                        "ovos.voiceid.user_info.response",
                        {
                            "user_id": None,
                            "status": "error",
                            "message": "User ID is required",
                        },
                    )
                )
            return

        if self.voice_processor is None:
            if self.bus:
                self.bus.emit(
                    Message(
                        "ovos.voiceid.user_info.response",
                        {
                            "user_id": user_id,
                            "status": "error",
                            "message": "Voice processor not initialized",
                        },
                    )
                )
            return

        try:
            # Check if user exists
            enrolled_users = self.voice_processor.get_enrolled_users()

            if user_id not in enrolled_users:
                if self.bus:
                    self.bus.emit(
                        Message(
                            "ovos.voiceid.user_info.response",
                            {
                                "user_id": user_id,
                                "status": "not_found",
                                "message": f"User {user_id} not found in enrolled users",
                                "available_users": enrolled_users,
                            },
                        )
                    )
                return

            # Get user information
            model_info = self.voice_processor.get_model_info()

            user_info = {
                "user_id": user_id,
                "is_enrolled": True,
                "enrollment_date": "unknown",  # Could be enhanced if we track this
                "model_source": model_info.get("model_source"),
                "confidence_threshold": model_info.get("confidence_threshold"),
                "sample_rate": model_info.get("sample_rate"),
            }

            # Could add more detailed info if available from voice processor
            if (
                hasattr(self.voice_processor, "user_embeddings")
                and user_id in self.voice_processor.user_embeddings
            ):
                embedding = self.voice_processor.user_embeddings[user_id]
                if hasattr(embedding, "shape"):
                    user_info["embedding_dimensions"] = embedding.shape
                elif isinstance(embedding, (list, tuple)):
                    user_info["embedding_dimensions"] = len(embedding)

            LOG.info(f"Retrieved information for user {user_id}")

            if self.bus:
                self.bus.emit(
                    Message(
                        "ovos.voiceid.user_info.response",
                        {
                            "user_id": user_id,
                            "status": "success",
                            "user_info": user_info,
                            "message": f"Information retrieved for user {user_id}",
                        },
                    )
                )

        except Exception as e:
            LOG.error(f"Failed to get user info for {user_id}: {e}")
            if self.bus:
                self.bus.emit(
                    Message(
                        "ovos.voiceid.user_info.response",
                        {
                            "user_id": user_id,
                            "status": "error",
                            "message": f"Failed to retrieve user info: {str(e)}",
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
            # Suppress logging during destruction to avoid reentrant call issues
            if hasattr(self, "voice_processor") and self.voice_processor:
                self.voice_processor.cleanup()
        except Exception:
            pass  # Suppress all cleanup errors during destruction


# Default configuration for OMVA Voice ID Plugin
OMVAVoiceIDConfig = {
    "model": "speechbrain/spkrec-ecapa-voxceleb",
    "confidence_threshold": 0.8,
    "sample_rate": 16000,
    "gpu": False,
    "enable_enrollment": True,
    "processing_timeout_ms": 100,
    "model_cache_dir": "/tmp/omva_models",  # Default cache directory
}


def launch_cli():
    """
    Launch the OMVA Voice ID plugin as a standalone service.
    This enables container/microservice deployment.
    """
    from ovos_utils import wait_for_exit_signal
    from ovos_bus_client.util import get_mycroft_bus
    from ovos_utils.log import init_service_logger
    from ovos_config import Configuration

    init_service_logger("omva-voiceid")
    LOG.info("Starting OMVA Voice ID standalone service...")

    # Load configuration from OVOS config system
    ovos_config = Configuration()
    plugin_config = ovos_config.get("audio_transformers", {}).get(
        "ovos-audio-transformer-plugin-omva-voiceid", {}
    )

    # Create plugin instance with configuration from OVOS
    config = OMVAVoiceIDConfig.copy()
    config.update(plugin_config)

    # Ensure essential settings
    config.update(
        {
            "confidence_threshold": config.get("confidence_threshold", 0.8),
            "enable_enrollment": config.get("enable_enrollment", True),
            "processing_timeout_ms": config.get("processing_timeout_ms", 100),
        }
    )

    LOG.info(f"Loaded configuration: model_cache_dir={config.get('model_cache_dir')}")
    LOG.info(f"Configuration keys: {list(config.keys())}")

    plugin = OMVAVoiceIDPlugin(config)

    # Connect to message bus
    bus = get_mycroft_bus()
    plugin.bind(bus)

    LOG.info("OMVA Voice ID service ready. Waiting for audio...")

    try:
        wait_for_exit_signal()  # Wait for Ctrl+C
    except KeyboardInterrupt:
        LOG.info("Shutdown signal received")
    finally:
        LOG.info("Shutting down OMVA Voice ID service...")
        plugin.shutdown()


if __name__ == "__main__":
    launch_cli()
