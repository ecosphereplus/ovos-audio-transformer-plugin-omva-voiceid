#!/usr/bin/env python3
"""
Test calibration effectiveness with realistic scenarios.
"""

import torch
import numpy as np
from ovos_audio_transformer_plugin_omva_voiceid.voice_processor import (
    OMVAVoiceProcessor,
)


def test_calibration_behavior():
    """
    Test the calibration function directly with different raw scores
    """
    print("=== CALIBRATION BEHAVIOR TEST ===")

    # Create processor instance
    config = {
        "confidence_threshold": 0.55,
        "model_cache_dir": "/tmp/test_cal_models",
        "sample_rate": 16000,
    }

    processor = OMVAVoiceProcessor(config)

    # Test calibration with different raw scores
    print("\n1. Testing calibration with various raw scores...")
    print("   Raw Score -> Calibrated Score")
    print("   " + "=" * 30)

    test_scores = [0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]
    num_users = 2  # Simulate 2 enrolled users

    calibrated_scores = []
    for raw_score in test_scores:
        calibrated = processor._calibrate_confidence(raw_score, num_users)
        calibrated_scores.append(calibrated)
        print(f"   {raw_score:.2f}        -> {calibrated:.3f} ({calibrated:.1%})")

    # Analyze calibration behavior
    print("\n2. Calibration Analysis...")

    # Find scores that reach 85%+ target
    high_confidence_threshold = 0.85
    qualifying_scores = [
        (raw, cal)
        for raw, cal in zip(test_scores, calibrated_scores)
        if cal >= high_confidence_threshold
    ]

    if qualifying_scores:
        print(f"   ✓ {len(qualifying_scores)} raw scores achieve 85%+ confidence:")
        for raw, cal in qualifying_scores:
            print(f"     Raw {raw:.2f} -> {cal:.1%}")

        min_raw_for_target = min(raw for raw, _ in qualifying_scores)
        print(f"   • Minimum raw score for 85%+: {min_raw_for_target:.2f}")
    else:
        print(f"   ❌ No raw scores achieve 85%+ confidence target")
        max_cal = max(calibrated_scores)
        print(f"   • Maximum calibrated score: {max_cal:.1%}")

    # Check discrimination capability
    low_scores = [
        cal for raw, cal in zip(test_scores, calibrated_scores) if raw <= 0.35
    ]
    high_scores = [
        cal for raw, cal in zip(test_scores, calibrated_scores) if raw >= 0.6
    ]

    if low_scores and high_scores:
        avg_low = np.mean(low_scores)
        avg_high = np.mean(high_scores)
        discrimination_ratio = avg_high / max(float(avg_low), 0.001)

        print(f"   • Discrimination analysis:")
        print(f"     Low raw scores (≤0.35) avg: {avg_low:.1%}")
        print(f"     High raw scores (≥0.6) avg: {avg_high:.1%}")
        print(f"     Discrimination ratio: {discrimination_ratio:.2f}x")

        if discrimination_ratio >= 2.0:
            print(f"   ✓ Discrimination target achieved (≥2.0x)")
        else:
            print(
                f"   ⚠ Discrimination below target ({discrimination_ratio:.2f}x < 2.0x)"
            )

    # Test with different user counts
    print("\n3. Multi-user impact analysis...")
    raw_score = 0.7  # Good raw score
    for users in [1, 2, 3, 5, 10]:
        calibrated = processor._calibrate_confidence(raw_score, users)
        print(f"   {users} users: Raw {raw_score:.2f} -> {calibrated:.1%}")

    return len(qualifying_scores) > 0


def test_realistic_similarity_scores():
    """
    Test calibration with realistic similarity scores from SpeechBrain
    """
    print("\n\n=== REALISTIC SCORE SIMULATION ===")

    config = {
        "confidence_threshold": 0.55,
        "model_cache_dir": "/tmp/test_real_models",
        "sample_rate": 16000,
    }

    processor = OMVAVoiceProcessor(config)

    # Simulate realistic scenarios based on SpeechBrain behavior
    scenarios = {
        "Same speaker (excellent match)": [0.85, 0.88, 0.82, 0.87, 0.90],
        "Same speaker (good match)": [0.75, 0.78, 0.72, 0.80, 0.76],
        "Same speaker (moderate match)": [0.65, 0.68, 0.62, 0.70, 0.66],
        "Different speaker (similar voice)": [0.45, 0.48, 0.42, 0.50, 0.46],
        "Different speaker (different voice)": [0.25, 0.28, 0.22, 0.30, 0.26],
        "Unknown/noise": [0.15, 0.18, 0.12, 0.20, 0.16],
    }

    print("Scenario Analysis:")
    print("=" * 60)

    results = {}
    for scenario, raw_scores in scenarios.items():
        calibrated_scores = [
            processor._calibrate_confidence(raw, 2) for raw in raw_scores
        ]
        avg_calibrated = np.mean(calibrated_scores)
        results[scenario] = avg_calibrated

        print(f"{scenario}:")
        print(f"  Raw scores: {raw_scores}")
        print(f"  Calibrated: {[f'{c:.2f}' for c in calibrated_scores]}")
        print(f"  Average: {avg_calibrated:.1%}")
        print()

    # Performance evaluation
    print("Performance Evaluation:")
    print("=" * 30)

    same_speaker_scores = [
        results["Same speaker (excellent match)"],
        results["Same speaker (good match)"],
        results["Same speaker (moderate match)"],
    ]

    different_speaker_scores = [
        results["Different speaker (similar voice)"],
        results["Different speaker (different voice)"],
        results["Unknown/noise"],
    ]

    min_same = min(same_speaker_scores)
    max_different = max(different_speaker_scores)

    print(f"Best same speaker performance: {max(same_speaker_scores):.1%}")
    print(f"Worst same speaker performance: {min_same:.1%}")
    print(f"Best different speaker performance: {max_different:.1%}")
    print(f"Worst different speaker performance: {min(different_speaker_scores):.1%}")

    confidence_target_met = min_same >= 0.85
    discrimination_achieved = (
        min_same / max_different >= 2.0 if max_different > 0 else True
    )

    print(f"\n🎯 Target Achievement:")
    print(
        f"   85%+ Confidence: {'✓ ACHIEVED' if confidence_target_met else '❌ MISSED'} ({min_same:.1%})"
    )
    print(
        f"   2.0x+ Discrimination: {'✓ ACHIEVED' if discrimination_achieved else '❌ MISSED'} ({min_same/max_different:.2f}x)"
    )

    return confidence_target_met and discrimination_achieved


if __name__ == "__main__":
    print("🔧 Testing Enhanced Confidence Calibration System\n")

    try:
        # Test calibration behavior
        calibration_success = test_calibration_behavior()

        # Test with realistic scenarios
        scenario_success = test_realistic_similarity_scores()

        # Overall result
        print(f"\n{'='*50}")
        print(
            f"OVERALL RESULT: {'SUCCESS' if calibration_success and scenario_success else 'NEEDS TUNING'}"
        )
        print(f"{'='*50}")

        if calibration_success and scenario_success:
            print("✅ Calibration system successfully achieves:")
            print("   • 85%+ confidence for legitimate users")
            print("   • 2.0x+ discrimination between users and non-users")
            print("   • Appropriate multi-user scaling")
        else:
            print("⚠️  Calibration system needs further tuning:")
            if not calibration_success:
                print("   • Raw score calibration insufficient")
            if not scenario_success:
                print("   • Realistic scenario performance inadequate")

        exit(0 if (calibration_success and scenario_success) else 1)

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
