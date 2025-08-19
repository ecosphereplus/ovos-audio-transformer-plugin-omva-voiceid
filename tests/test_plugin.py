"""Unit tests for OMVA Voice Identification Plugin"""

import unittest
from unittest.mock import Mock, patch
from ovos_audio_transformer_plugin_omva_voiceid import OMVAVoiceIDPlugin


class TestOMVAVoiceIDPlugin(unittest.TestCase):
    def setUp(self):
        self.config = {"confidence_threshold": 0.8}
        
    def test_plugin_initialization(self):
        plugin = OMVAVoiceIDPlugin(self.config)
        self.assertEqual(plugin.confidence_threshold, 0.8)
        
    def test_transform_passthrough(self):
        plugin = OMVAVoiceIDPlugin(self.config)
        test_audio = b"test_audio_data"
        result = plugin.transform(test_audio)
        self.assertEqual(result, test_audio)


if __name__ == '__main__':
    unittest.main()
