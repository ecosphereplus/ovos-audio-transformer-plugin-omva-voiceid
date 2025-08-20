"""
Unit tests for OMVA Voice Identification Plugin with SpeechBrain integration
"""

import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from ovos_audio_transformer_plugin_omva_voiceid import OMVAVoiceIDPlugin


class TestOMVAVoiceIDPlugin(unittest.TestCase):
    """Test cases for the main plugin class"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            "confidence_threshold": 0.25,
            "enable_enrollment": True,
            "processing_timeout_ms": 200,
            "voice_processing": {
                "sample_rate": 16000,
                "model_source": "speechbrain/spkrec-ecapa-voxceleb",
                "verification_threshold": 0.25,
            },
        }

    @patch("ovos_audio_transformer_plugin_omva_voiceid.OMVAVoiceProcessor")
    def test_plugin_initialization(self, mock_processor_class):
        """Test plugin initialization with SpeechBrain configuration"""
        mock_processor = Mock()
        mock_processor.get_model_info.return_value = {
            "model_available": True,
            "model_source": "speechbrain/spkrec-ecapa-voxceleb",
            "enrolled_users": 0,
        }
        mock_processor_class.return_value = mock_processor

        plugin = OMVAVoiceIDPlugin(self.config)

        self.assertEqual(plugin.confidence_threshold, 0.25)
        self.assertTrue(plugin.enable_enrollment)
        self.assertEqual(plugin.processing_timeout_ms, 200)
        self.assertIsNotNone(plugin.voice_processor)
        mock_processor_class.assert_called_once()

    def test_audiochunk2array_conversion(self):
        """Test audio chunk to PyTorch tensor conversion"""
        # Create test audio data (16-bit PCM)
        test_audio = np.array([100, -200, 300, -400], dtype=np.int16).tobytes()

        result = OMVAVoiceIDPlugin.audiochunk2array(test_audio)

        # Check result is PyTorch tensor
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(len(result), 4)

        # Check normalization (should be between -1 and 1)
        self.assertTrue(torch.all(result >= -1.0))
        self.assertTrue(torch.all(result <= 1.0))

    @patch("ovos_audio_transformer_plugin_omva_voiceid.OMVAVoiceProcessor")
    def test_identify_speaker_success(self, mock_processor_class):
        """Test successful speaker identification with SpeechBrain"""
        # Mock voice processor
        mock_processor = Mock()
        mock_processor.get_model_info.return_value = {"model_available": True}
        mock_processor.identify_speaker.return_value = ("john_doe", 0.87)
        mock_processor_class.return_value = mock_processor

        plugin = OMVAVoiceIDPlugin(self.config)

        # Create test audio tensor
        test_audio = np.random.randint(-1000, 1000, size=8000, dtype=np.int16).tobytes()

        speaker_id, confidence = plugin.identify_speaker(test_audio)

        self.assertEqual(speaker_id, "john_doe")
        self.assertEqual(confidence, 0.87)

        # Check that processor was called with tensor
        mock_processor.identify_speaker.assert_called_once()
        call_args = mock_processor.identify_speaker.call_args[0][0]
        self.assertIsInstance(call_args, torch.Tensor)

    @patch("ovos_audio_transformer_plugin_omva_voiceid.OMVAVoiceProcessor")
    def test_identify_speaker_no_processor(self, mock_processor_class):
        """Test speaker identification when voice processor is not initialized"""
        plugin = OMVAVoiceIDPlugin(self.config)
        plugin.voice_processor = None  # Simulate failed initialization

        test_audio = np.array([100, -200, 300], dtype=np.int16).tobytes()
        speaker_id, confidence = plugin.identify_speaker(test_audio)

        self.assertIsNone(speaker_id)
        self.assertEqual(confidence, 0.0)

    @patch("ovos_audio_transformer_plugin_omva_voiceid.OMVAVoiceProcessor")
    def test_identify_speaker_success_confidence(self, mock_processor_class):
        """Test successful speaker identification"""
        # Mock voice processor
        mock_processor = Mock()
        mock_processor.identify_speaker.return_value = ("john_doe", 0.87)
        mock_processor_class.return_value = mock_processor

        plugin = OMVAVoiceIDPlugin(self.config)

        # Create test audio (sufficient length)
        test_audio = np.random.randint(-1000, 1000, size=8000, dtype=np.int16).tobytes()

        speaker_id, confidence = plugin.identify_speaker(test_audio)

        self.assertEqual(speaker_id, "john_doe")
        self.assertEqual(confidence, 0.87)

        # Check that processor was called
        mock_processor.identify_speaker.assert_called_once()

    @patch("ovos_audio_transformer_plugin_omva_voiceid.OMVAVoiceProcessor")
    def test_identify_speaker_short_audio(self, mock_processor_class):
        """Test speaker identification with too short audio"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor

        plugin = OMVAVoiceIDPlugin(self.config)

        # Create very short audio
        test_audio = np.array([100, -200], dtype=np.int16).tobytes()

        speaker_id, confidence = plugin.identify_speaker(test_audio)

        self.assertIsNone(speaker_id)
        self.assertEqual(confidence, 0.0)

        # Processor should not be called for short audio
        mock_processor.identify_speaker.assert_not_called()

    @patch("ovos_audio_transformer_plugin_omva_voiceid.OMVAVoiceProcessor")
    def test_transform_pass_through(self, mock_processor_class):
        """Test that transform method passes audio through unchanged"""
        mock_processor = Mock()
        mock_processor.identify_speaker.return_value = ("jane_doe", 0.92)
        mock_processor_class.return_value = mock_processor

        plugin = OMVAVoiceIDPlugin(self.config)
        plugin.bus = Mock()  # Mock message bus

        # Test with raw bytes
        test_audio = np.random.randint(-1000, 1000, size=8000, dtype=np.int16).tobytes()
        result = plugin.transform(test_audio)

        # Audio should pass through unchanged
        self.assertEqual(result, test_audio)

        # Message should be emitted
        plugin.bus.emit.assert_called_once()

        # Check message type and content
        call_args = plugin.bus.emit.call_args[0][0]
        self.assertEqual(call_args.msg_type, "ovos.voice.identified")
        self.assertEqual(call_args.data["speaker_id"], "jane_doe")
        self.assertEqual(call_args.data["confidence"], 0.92)

    @patch("ovos_audio_transformer_plugin_omva_voiceid.OMVAVoiceProcessor")
    def test_transform_unknown_speaker(self, mock_processor_class):
        """Test transform with unknown speaker (low confidence)"""
        mock_processor = Mock()
        mock_processor.identify_speaker.return_value = (
            "john_doe",
            0.3,
        )  # Low confidence
        mock_processor_class.return_value = mock_processor

        plugin = OMVAVoiceIDPlugin(self.config)
        plugin.bus = Mock()

        test_audio = np.random.randint(-1000, 1000, size=8000, dtype=np.int16).tobytes()
        result = plugin.transform(test_audio)

        # Audio should still pass through
        self.assertEqual(result, test_audio)

        # Should emit unknown event
        plugin.bus.emit.assert_called_once()
        call_args = plugin.bus.emit.call_args[0][0]
        self.assertEqual(call_args.msg_type, "ovos.voice.unknown")

    @patch("ovos_audio_transformer_plugin_omva_voiceid.OMVAVoiceProcessor")
    def test_message_bus_binding(self, mock_processor_class):
        """Test message bus event handler registration"""
        mock_processor_class.return_value = Mock()

        plugin = OMVAVoiceIDPlugin(self.config)
        mock_bus = Mock()

        plugin.bind(mock_bus)

        # Check that event handlers are registered
        expected_calls = [
            ("ovos.voiceid.get_stats", plugin.handle_get_stats),
            ("ovos.voiceid.reset_stats", plugin.handle_reset_stats),
            ("ovos.voiceid.enroll_user", plugin.handle_enroll_user),
            ("ovos.voiceid.list_users", plugin.handle_list_users),
        ]

        for event_type, handler in expected_calls:
            mock_bus.on.assert_any_call(event_type, handler)

    @patch("ovos_audio_transformer_plugin_omva_voiceid.OMVAVoiceProcessor")
    def test_statistics_handling(self, mock_processor_class):
        """Test statistics request handling"""
        mock_processor_class.return_value = Mock()

        plugin = OMVAVoiceIDPlugin(self.config)
        plugin.bus = Mock()

        # Test get stats
        message = Mock()
        plugin.handle_get_stats(message)

        # Should emit stats response
        plugin.bus.emit.assert_called()
        call_args = plugin.bus.emit.call_args[0][0]
        self.assertEqual(call_args.msg_type, "ovos.voiceid.stats.response")
        self.assertIn("total_processed", call_args.data)

        # Test reset stats
        plugin.bus.reset_mock()
        plugin.handle_reset_stats(message)

        # Should emit reset confirmation
        plugin.bus.emit.assert_called()
        call_args = plugin.bus.emit.call_args[0][0]
        self.assertEqual(call_args.msg_type, "ovos.voiceid.stats.reset")
        self.assertEqual(call_args.data["status"], "success")


class TestVoiceProcessor(unittest.TestCase):
    """Test cases for voice processor (if available)"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            "sample_rate": 16000,
            "mfcc_coefficients": 13,
            "window_size": 0.025,
            "hop_length": 0.01,
            "cache_dir": "/tmp/test_omva_voiceid",
        }

    def test_voice_processor_import(self):
        """Test that voice processor can be imported"""
        try:
            from ovos_audio_transformer_plugin_omva_voiceid.voice_processor import (
                OMVAVoiceProcessor,
            )

            # If we get here, import succeeded
            self.assertTrue(True)
        except ImportError as e:
            # Expected if dependencies are not installed
            self.skipTest(f"Voice processor dependencies not available: {e}")

    # @patch("ovos_audio_transformer_plugin_omva_voiceid.voice_processor.librosa")
    # @patch("ovos_audio_transformer_plugin_omva_voiceid.voice_processor.SVC")
    # @patch("ovos_audio_transformer_plugin_omva_voiceid.voice_processor.StandardScaler")
    def test_voice_processor_initialization(self):
        """Test voice processor initialization"""
        from ovos_audio_transformer_plugin_omva_voiceid.voice_processor import (
            OMVAVoiceProcessor,
        )

        processor = OMVAVoiceProcessor(self.config)

        self.assertEqual(processor.sample_rate, 16000)


if __name__ == "__main__":
    unittest.main()
