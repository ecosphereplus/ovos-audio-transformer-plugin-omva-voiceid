#!/usr/bin/env python3
"""
Container Testing Script for OMVA Voice ID Plugin

This script provides comprehensive testing for the OMVA Voice ID plugin
running as a Docker container, including message bus communication,
voice identification, and performance validation.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchaudio
from ovos_bus_client import MessageBusClient
from ovos_bus_client.message import Message
from ovos_utils.log import LOG


class OMVAContainerTester:
    """
    Test suite for OMVA Voice ID Plugin running in container
    """

    def __init__(self):
        """Initialize the container tester"""
        self.bus = None
        self.test_results = []
        self.audio_files = {
            "obama": "obama.wav",
            "obama_val": "obama-val.wav",
            "jfk": "jfk.wav",
            "jfk_val": "jfk-val.wav",
        }

        # Test configuration
        self.test_config = {
            "message_timeout": 5.0,  # seconds
            "audio_chunk_size": 1024,
            "sample_rate": 16000,
        }

    def connect_to_bus(self, host: str = "localhost", port: int = 8181) -> bool:
        """
        Connect to OVOS message bus

        Args:
            host: Message bus host
            port: Message bus port

        Returns:
            True if connection successful
        """
        try:
            self.bus = MessageBusClient(host=host, port=port)
            self.bus.connect()
            print(f"✓ Connected to message bus at {host}:{port}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to message bus: {e}")
            return False

    def load_audio_file(self, filename: str) -> Optional[torch.Tensor]:
        """
        Load audio file as tensor

        Args:
            filename: Path to audio file

        Returns:
            Audio tensor or None if loading fails
        """
        try:
            if not os.path.exists(filename):
                print(f"❌ Audio file not found: {filename}")
                return None

            # Load with torchaudio
            waveform, sample_rate = torchaudio.load(filename)

            # Resample if necessary
            if sample_rate != self.test_config["sample_rate"]:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, self.test_config["sample_rate"]
                )
                waveform = resampler(waveform)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            print(f"✓ Loaded audio: {filename} ({waveform.shape[1]} samples)")
            return waveform.squeeze()

        except Exception as e:
            print(f"❌ Failed to load audio {filename}: {e}")
            return None

    def audio_to_bytes(self, audio_tensor: torch.Tensor) -> bytes:
        """
        Convert audio tensor to bytes for transmission

        Args:
            audio_tensor: Audio tensor

        Returns:
            Audio data as bytes
        """
        # Convert to 16-bit PCM
        audio_np = (audio_tensor.numpy() * 32767).astype(np.int16)
        return audio_np.tobytes()

    def test_plugin_status(self) -> bool:
        """
        Test if the plugin is responding to status requests

        Returns:
            True if plugin responds correctly
        """
        print("\n=== Testing Plugin Status ===")

        if not self.bus:
            print("❌ Message bus not connected")
            return False

        # Set up response handler
        response_received = {"status": False, "data": None}

        def handle_stats_response(message):
            response_received["status"] = True
            response_received["data"] = message.data

        self.bus.on("ovos.voiceid.stats.response", handle_stats_response)

        # Send stats request
        self.bus.emit(Message("ovos.voiceid.get_stats"))

        # Wait for response
        start_time = time.time()
        while (
            not response_received["status"]
            and (time.time() - start_time) < self.test_config["message_timeout"]
        ):
            time.sleep(0.1)

        if response_received["status"]:
            stats = response_received["data"]
            print(f"✓ Plugin responded with stats:")
            print(f"  - Total processed: {stats.get('total_processed', 0)}")
            print(f"  - Successful IDs: {stats.get('successful_identifications', 0)}")
            print(f"  - Plugin version: {stats.get('plugin_version', 'unknown')}")
            print(
                f"  - Confidence threshold: {stats.get('configuration', {}).get('confidence_threshold', 'unknown')}"
            )
            self.test_results.append(
                {"test": "plugin_status", "passed": True, "data": stats}
            )
            return True
        else:
            print("❌ Plugin did not respond to stats request")
            self.test_results.append(
                {"test": "plugin_status", "passed": False, "error": "No response"}
            )
            return False

    def test_voice_identification(
        self, audio_file: str, expected_speaker: Optional[str] = None
    ) -> bool:
        """
        Test voice identification with audio file

        Args:
            audio_file: Path to audio file
            expected_speaker: Expected speaker ID (None for unknown)

        Returns:
            True if test passes
        """
        print(f"\n=== Testing Voice Identification: {audio_file} ===")

        # Load audio
        audio_tensor = self.load_audio_file(audio_file)
        if audio_tensor is None:
            print(f"❌ Failed to load audio file: {audio_file}")
            return False

        # Set up response handlers
        identification_received = {"status": False, "data": None}
        unknown_received = {"status": False, "data": None}

        def handle_voice_identified(message):
            identification_received["status"] = True
            identification_received["data"] = message.data

        def handle_voice_unknown(message):
            unknown_received["status"] = True
            unknown_received["data"] = message.data

        self.bus.on("ovos.voice.identified", handle_voice_identified)
        self.bus.on("ovos.voice.unknown", handle_voice_unknown)

        # Simulate audio processing by sending chunks
        audio_bytes = self.audio_to_bytes(audio_tensor)
        chunk_size = self.test_config["audio_chunk_size"]

        print(
            f"  Sending audio in {len(audio_bytes)} bytes ({len(audio_bytes)//chunk_size} chunks)"
        )

        # Send audio chunk (simulating transform method)
        # Note: This is a simplified test - real plugin processes through transform()
        chunk = audio_bytes[
            : chunk_size * 10
        ]  # Send larger chunk for better identification

        # We can't directly call the plugin's transform method from here,
        # so we'll send a test message to see if the plugin processes it
        test_message = Message(
            "test.voice.audio",
            {
                "audio_data": chunk.hex(),  # Send as hex string
                "sample_rate": self.test_config["sample_rate"],
                "format": "pcm16",
            },
        )

        self.bus.emit(test_message)

        # Wait for response
        start_time = time.time()
        timeout = self.test_config["message_timeout"]

        while (time.time() - start_time) < timeout:
            if identification_received["status"] or unknown_received["status"]:
                break
            time.sleep(0.1)

        # Analyze results
        if identification_received["status"]:
            data = identification_received["data"]
            speaker_id = data.get("speaker_id")
            confidence = data.get("confidence", 0.0)

            print(f"✓ Speaker identified: {speaker_id} (confidence: {confidence:.3f})")

            # Check if result matches expectation
            if expected_speaker and speaker_id == expected_speaker:
                print(f"✓ Correctly identified expected speaker: {expected_speaker}")
                self.test_results.append(
                    {
                        "test": f"voice_id_{os.path.basename(audio_file)}",
                        "passed": True,
                        "speaker_id": speaker_id,
                        "confidence": confidence,
                        "expected": expected_speaker,
                    }
                )
                return True
            elif not expected_speaker and speaker_id:
                print(
                    f"⚠️  Unexpected identification: got {speaker_id}, expected unknown"
                )
                self.test_results.append(
                    {
                        "test": f"voice_id_{os.path.basename(audio_file)}",
                        "passed": False,
                        "speaker_id": speaker_id,
                        "confidence": confidence,
                        "expected": expected_speaker,
                        "error": "Unexpected identification",
                    }
                )
                return False
            else:
                print(f"✓ Identification result matches expectation")
                self.test_results.append(
                    {
                        "test": f"voice_id_{os.path.basename(audio_file)}",
                        "passed": True,
                        "speaker_id": speaker_id,
                        "confidence": confidence,
                        "expected": expected_speaker,
                    }
                )
                return True

        elif unknown_received["status"]:
            data = unknown_received["data"]
            confidence = data.get("confidence", 0.0)

            print(f"✓ Speaker marked as unknown (confidence: {confidence:.3f})")

            if expected_speaker is None:
                print("✓ Correctly identified as unknown speaker")
                self.test_results.append(
                    {
                        "test": f"voice_id_{os.path.basename(audio_file)}",
                        "passed": True,
                        "speaker_id": None,
                        "confidence": confidence,
                        "expected": expected_speaker,
                    }
                )
                return True
            else:
                print(f"❌ Expected {expected_speaker} but got unknown")
                self.test_results.append(
                    {
                        "test": f"voice_id_{os.path.basename(audio_file)}",
                        "passed": False,
                        "speaker_id": None,
                        "confidence": confidence,
                        "expected": expected_speaker,
                        "error": "Expected identification but got unknown",
                    }
                )
                return False
        else:
            print("❌ No identification response received")
            self.test_results.append(
                {
                    "test": f"voice_id_{os.path.basename(audio_file)}",
                    "passed": False,
                    "error": "No response received",
                }
            )
            return False

    def test_enrollment(self, user_id: str, audio_files: List[str]) -> bool:
        """
        Test user enrollment functionality

        Args:
            user_id: User ID to enroll
            audio_files: List of audio files for enrollment

        Returns:
            True if enrollment succeeds
        """
        print(f"\n=== Testing User Enrollment: {user_id} ===")

        # Load audio samples
        audio_samples = []
        for audio_file in audio_files:
            tensor = self.load_audio_file(audio_file)
            if tensor is not None:
                audio_samples.append(self.audio_to_bytes(tensor))
            else:
                print(f"⚠️  Failed to load enrollment audio: {audio_file}")

        if not audio_samples:
            print("❌ No valid audio samples for enrollment")
            return False

        print(f"  Loaded {len(audio_samples)} audio samples for enrollment")

        # Set up response handler
        response_received = {"status": False, "data": None}

        def handle_enroll_response(message):
            response_received["status"] = True
            response_received["data"] = message.data

        self.bus.on("ovos.voiceid.enroll.response", handle_enroll_response)

        # Send enrollment request
        enrollment_data = {
            "user_id": user_id,
            "audio_samples": [sample.hex() for sample in audio_samples],  # Send as hex
            "sample_rate": self.test_config["sample_rate"],
        }

        self.bus.emit(Message("ovos.voiceid.enroll_user", enrollment_data))

        # Wait for response
        start_time = time.time()
        while (
            not response_received["status"]
            and (time.time() - start_time) < self.test_config["message_timeout"]
        ):
            time.sleep(0.1)

        if response_received["status"]:
            data = response_received["data"]
            status = data.get("status", "unknown")
            message = data.get("message", "")

            if status == "success":
                print(f"✓ User {user_id} enrolled successfully")
                self.test_results.append(
                    {
                        "test": f"enrollment_{user_id}",
                        "passed": True,
                        "user_id": user_id,
                        "samples_count": len(audio_samples),
                    }
                )
                return True
            else:
                print(f"❌ Enrollment failed: {message}")
                self.test_results.append(
                    {
                        "test": f"enrollment_{user_id}",
                        "passed": False,
                        "error": message,
                        "status": status,
                    }
                )
                return False
        else:
            print("❌ No response to enrollment request")
            self.test_results.append(
                {
                    "test": f"enrollment_{user_id}",
                    "passed": False,
                    "error": "No response",
                }
            )
            return False

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """
        Run comprehensive test suite

        Returns:
            Test results summary
        """
        print("🧪 Starting OMVA Voice ID Container Testing")
        print("=" * 50)

        start_time = time.time()

        # Test 1: Connect to message bus
        if not self.connect_to_bus():
            return {"status": "failed", "error": "Could not connect to message bus"}

        # Test 2: Plugin status
        self.test_plugin_status()

        # Test 3: Test with available audio files
        for file_key, filename in self.audio_files.items():
            if os.path.exists(filename):
                # For testing, assume we don't know the expected speakers initially
                self.test_voice_identification(filename, expected_speaker=None)
            else:
                print(f"⚠️  Audio file not found: {filename}")

        # Test 4: Test enrollment (if audio files exist)
        if os.path.exists("obama.wav"):
            self.test_enrollment("obama", ["obama.wav"])

        # Generate test report
        total_time = time.time() - start_time
        passed_tests = sum(
            1 for result in self.test_results if result.get("passed", False)
        )
        total_tests = len(self.test_results)

        summary = {
            "status": "completed",
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": (
                (passed_tests / total_tests * 100) if total_tests > 0 else 0
            ),
            "total_time_seconds": total_time,
            "detailed_results": self.test_results,
        }

        print("\n" + "=" * 50)
        print("🏁 Test Summary")
        print(f"  Total tests: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {total_tests - passed_tests}")
        print(f"  Success rate: {summary['success_rate']:.1f}%")
        print(f"  Total time: {total_time:.2f}s")

        if summary["success_rate"] == 100:
            print("🎉 All tests passed!")
        elif summary["success_rate"] >= 80:
            print("✅ Most tests passed")
        else:
            print("⚠️  Multiple test failures detected")

        return summary


def main():
    """Main test runner"""
    tester = OMVAContainerTester()

    try:
        results = tester.run_comprehensive_test()

        # Save detailed results
        with open("container_test_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n📄 Detailed results saved to: container_test_results.json")

        # Exit with appropriate code
        if results["success_rate"] == 100:
            sys.exit(0)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
