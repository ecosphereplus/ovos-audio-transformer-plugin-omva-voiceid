"""
OMVA Voice Identification Plugin for OVOS

This plugin provides multi-user voice identification capabilities for OVOS.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from ovos_bus_client.message import Message
from ovos_plugin_manager.templates.transformers import AudioTransformer
from ovos_utils.log import LOG
from speech_recognition import AudioData

from .version import VERSION_BUILD, VERSION_MAJOR, VERSION_MINOR

__version__ = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_BUILD}"


class OMVAVoiceIDPlugin(AudioTransformer):
    """OMVA Voice Identification AudioTransformer Plugin"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        super().__init__("ovos-audio-transformer-plugin-omva-voiceid", 10, config)

        self.confidence_threshold = self.config.get("confidence_threshold", 0.8)
        self.processing_stats = {"total_processed": 0, "successful_ids": 0}

        LOG.info(f"OMVA Voice ID Plugin v{__version__} initialized")

    def identify_speaker(self, signal: torch.Tensor) -> Tuple[Optional[str], float]:
        """Identify speaker from audio data - placeholder implementation"""
        # TODO: Implement actual voice identification
        # For now, return placeholder values
        return "unknown", 0.0

    @staticmethod
    def audiochunk2array(audio_data):
        """Convert audio data bytes to a normalized float32 tensor"""
        # Ensure audio_data is bytes
        # Convert buffer to float32 using NumPy
        audio_as_np_int16 = np.frombuffer(audio_data, dtype=np.int16)
        audio_as_np_float32 = audio_as_np_int16.astype(np.float32)

        # Normalise float32 array so that values are between -1.0 and +1.0
        max_int16 = 2**15
        data = audio_as_np_float32 / max_int16
        return torch.from_numpy(data).float()

    def transform(self, audio_data) -> Any:
        """Main AudioTransformer method"""
        # Handle both raw bytes and AudioData objects
        if isinstance(audio_data, AudioData):
            raw_audio = audio_data.get_wav_data()
        else:
            raw_audio = audio_data

        # Skip very small audio chunks
        if len(raw_audio) < 1024:
            return audio_data

        signal = self.audiochunk2array(raw_audio)

        # Perform voice identification (placeholder)
        speaker_id, confidence = self.identify_speaker(signal)

        # Emit identification event
        if self.bus:
            event_data = {
                "speaker_id": speaker_id,
                "confidence": confidence,
                "plugin_version": __version__,
            }

            if confidence >= self.confidence_threshold:
                self.bus.emit(Message("ovos.voice.identified", event_data))
            else:
                self.bus.emit(Message("ovos.voice.unknown", event_data))

        # Update stats
        self.processing_stats["total_processed"] += 1
        if confidence >= self.confidence_threshold:
            self.processing_stats["successful_ids"] += 1

        # Return audio unchanged (pass-through)
        return audio_data

    def bind(self, bus=None):
        """Bind to message bus"""
        super().bind(bus)
        if self.bus:
            self.bus.on("ovos.voiceid.get_stats", self.handle_get_stats)
            LOG.info("OMVA Voice ID plugin bound to message bus")

    def handle_get_stats(self, message):
        """Handle statistics request"""
        if self.bus:
            self.bus.emit(
                Message(
                    "ovos.voiceid.stats.response",
                    {**self.processing_stats, "plugin_version": __version__},
                )
            )

    def shutdown(self):
        """Cleanup on shutdown"""
        LOG.info("OMVA Voice ID plugin shutting down")
