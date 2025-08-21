#!/usr/bin/env python3
"""
Comprehensive test to verify confidence achievement and discrimination improvements.
"""

import os
import numpy as np
import torch
import torchaudio
from ovos_audio_transformer_plugin_omva_voiceid.voice_processor import (
    OMVAVoiceProcessor,
)


def create_synthetic_voice_samples(
    user_id: str, base_freq: float = 220.0, duration: float = 3.0
):
    """
    Create synthetic voice samples for testing

    Args:
        user_id: Identifier for the user
        base_freq: Base frequency for voice simulation
        duration: Duration of audio in seconds

    Returns:
        List of PyTorch tensors representing audio samples
    """
    sample_rate = 16000
    num_samples = int(sample_rate * duration)

    # Create 5 different voice samples for enrollment
    samples = []
    for i in range(5):
        # Create slightly different frequency for each sample
        freq = base_freq + (i * 10)  # Add variation
        t = torch.linspace(0, duration, num_samples)

        # Create a more complex waveform
        signal = (
            0.3 * torch.sin(2 * np.pi * freq * t)  # Fundamental
            + 0.2 * torch.sin(2 * np.pi * freq * 2 * t)  # 2nd harmonic
            + 0.1 * torch.sin(2 * np.pi * freq * 3 * t)  # 3rd harmonic
            + 0.05 * torch.randn(num_samples)  # Add some noise for realism
        )

        # Add some amplitude modulation for more realism
        am_freq = 5.0  # Amplitude modulation frequency
        am_signal = 1.0 + 0.3 * torch.sin(2 * np.pi * am_freq * t)
        signal = signal * am_signal

        # Normalize
        signal = signal / torch.max(torch.abs(signal)) * 0.8

        samples.append(signal.unsqueeze(0))  # Add batch dimension

    return samples


def test_confidence_and_discrimination():
    """
    Test confidence achievement and discrimination ratio
    """
    print("=== CONFIDENCE & DISCRIMINATION ACHIEVEMENT TEST ===")

    # Initialize processor
    config = {
        "confidence_threshold": 0.25,  # Lower for more sensitive testing
        "model_cache_dir": "/tmp/test_confidence_models",
        "sample_rate": 16000,
        "gpu": False,
    }

    processor = OMVAVoiceProcessor(config)

    # Create synthetic voice samples for different users
    print("\n1. Creating synthetic voice samples...")
    obama_samples = create_synthetic_voice_samples(
        "obama", base_freq=150.0, duration=4.0
    )
    jfk_samples = create_synthetic_voice_samples("jfk", base_freq=180.0, duration=4.0)
    unknown_samples = create_synthetic_voice_samples(
        "unknown", base_freq=200.0, duration=4.0
    )

    print(f"   Created {len(obama_samples)} Obama samples")
    print(f"   Created {len(jfk_samples)} JFK samples")
    print(f"   Created {len(unknown_samples)} unknown samples")

    # Enroll users
    print("\n2. Enrolling users...")
    obama_success = processor.enroll_user(
        "obama", obama_samples[:3]
    )  # Use 3 samples for enrollment
    jfk_success = processor.enroll_user("jfk", jfk_samples[:3])

    print(f"   Obama enrollment: {'SUCCESS' if obama_success else 'FAILED'}")
    print(f"   JFK enrollment: {'SUCCESS' if jfk_success else 'FAILED'}")

    if not (obama_success and jfk_success):
        print("❌ Enrollment failed - cannot continue test")
        return False

    # Test recognition with remaining samples
    print("\n3. Testing recognition performance...")

    # Test Obama samples
    obama_confidences = []
    for i, sample in enumerate(obama_samples[3:]):  # Use unused samples
        speaker, confidence = processor.identify_speaker(sample)
        obama_confidences.append(confidence)
        speaker_name = speaker if speaker else "unknown"
        print(f"   Obama sample {i+4}: {speaker_name} (confidence: {confidence:.1%})")

    # Test JFK samples
    jfk_confidences = []
    for i, sample in enumerate(jfk_samples[3:]):
        speaker, confidence = processor.identify_speaker(sample)
        jfk_confidences.append(confidence)
        speaker_name = speaker if speaker else "unknown"
        print(f"   JFK sample {i+4}: {speaker_name} (confidence: {confidence:.1%})")

    # Test unknown user samples
    unknown_confidences = []
    unknown_correct = 0
    for i, sample in enumerate(unknown_samples[:2]):  # Test 2 unknown samples
        speaker, confidence = processor.identify_speaker(sample)
        unknown_confidences.append(confidence)
        speaker_name = speaker if speaker else "unknown"
        if speaker is None:  # None means unknown
            unknown_correct += 1
        print(f"   Unknown sample {i+1}: {speaker_name} (confidence: {confidence:.1%})")

    # Calculate performance metrics
    print("\n4. Performance Analysis...")

    obama_avg_conf = float(np.mean(obama_confidences)) if obama_confidences else 0.0
    jfk_avg_conf = float(np.mean(jfk_confidences)) if jfk_confidences else 0.0
    unknown_avg_conf = (
        float(np.mean(unknown_confidences)) if unknown_confidences else 0.0
    )

    print(f"   Obama average confidence: {obama_avg_conf:.1%}")
    print(f"   JFK average confidence: {jfk_avg_conf:.1%}")
    print(f"   Unknown average confidence: {unknown_avg_conf:.1%}")

    # Discrimination ratio
    min_legitimate_conf = float(min(obama_avg_conf, jfk_avg_conf))
    discrimination_ratio = min_legitimate_conf / max(
        unknown_avg_conf, 0.001
    )  # Avoid division by zero

    print(f"   Discrimination ratio: {discrimination_ratio:.2f}x")

    # Test results
    print("\n=== ACHIEVEMENT SUMMARY ===")

    confidence_target_met = min_legitimate_conf >= 0.85
    discrimination_target_met = discrimination_ratio >= 2.0
    security_maintained = (
        unknown_correct >= len(unknown_samples[:2]) * 0.5
    )  # At least 50% rejection

    print(
        f"✓ Confidence Target (85%+): {'ACHIEVED' if confidence_target_met else 'MISSED'} ({min_legitimate_conf:.1%})"
    )
    print(
        f"✓ Discrimination Target (2.0x+): {'ACHIEVED' if discrimination_target_met else 'MISSED'} ({discrimination_ratio:.2f}x)"
    )
    print(
        f"✓ Security Target (unknown rejection): {'MAINTAINED' if security_maintained else 'COMPROMISED'} ({unknown_correct}/{len(unknown_samples[:2])})"
    )

    # Overall success
    overall_success = (
        confidence_target_met and discrimination_target_met and security_maintained
    )
    print(
        f"\n🎯 OVERALL TARGET ACHIEVEMENT: {'SUCCESS' if overall_success else 'PARTIAL'}"
    )

    # Additional insights
    print(f"\n📊 DETAILED INSIGHTS:")
    print(
        f"   • Raw confidence boost: {((obama_avg_conf + jfk_avg_conf) / 2) / 0.45 * 100 - 100:.1f}% over typical raw scores"
    )
    print(
        f"   • Calibration effectiveness: {'HIGH' if min_legitimate_conf >= 0.85 else 'MODERATE' if min_legitimate_conf >= 0.70 else 'LOW'}"
    )
    print(
        f"   • Security robustness: {'STRONG' if unknown_avg_conf < 0.4 else 'MODERATE' if unknown_avg_conf < 0.6 else 'WEAK'}"
    )

    return overall_success


if __name__ == "__main__":
    try:
        success = test_confidence_and_discrimination()
        exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        exit(1)
