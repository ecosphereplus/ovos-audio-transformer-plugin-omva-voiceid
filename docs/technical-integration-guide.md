# OMVA Voice ID Plugin - Technical Integration Guide

## Overview

The OMVA Voice ID Plugin is a sophisticated speaker identification and verification system for the OpenVoiceOS ecosystem. It provides real-time voice identification using state-of-the-art deep learning models, with comprehensive message bus integration for seamless interaction with other OVOS plugins and skills.

## Table of Contents

1. [Architecture](#architecture)
2. [Message Bus API](#message-bus-api)
3. [Event System](#event-system)
4. [Integration Examples](#integration-examples)
5. [Audio Processing](#audio-processing)
6. [Error Handling](#error-handling)
7. [Performance Considerations](#performance-considerations)
8. [Container Integration](#container-integration)
9. [Testing and Validation](#testing-and-validation)

## Architecture

### Core Components

- **AudioTransformer Integration**: Processes audio streams in real-time
- **SpeechBrain ECAPA-TDNN Model**: State-of-the-art speaker recognition
- **Message Bus Interface**: Comprehensive API for external integration
- **User Management System**: Enrollment, verification, and user database
- **Event Emission System**: Real-time notifications for identification events

### Plugin Configuration

```json
{
    "listener": {
        "audio_transformers": {
            "ovos-audio-transformer-plugin-omva-voiceid": {
                "enabled": true,
                "model": "speechbrain/spkrec-ecapa-voxceleb",
                "model_cache_dir": "/opt/omva/speechbrain_cache",
                "sample_rate": 16000,
                "confidence_threshold": 0.8,
                "processing_timeout_ms": 100,
                "gpu": false,
                "enable_enrollment": true
            }
        }
    }
}
```

## Message Bus API

The plugin provides a comprehensive message bus API for integration with other OVOS components.

### Statistics and Monitoring

#### Get Plugin Statistics
```python
# Request
bus.emit(Message("ovos.voiceid.get_stats", {}))

# Response: ovos.voiceid.stats.response
{
    "total_processed": 1247,
    "successful_identifications": 892,
    "failed_identifications": 355,
    "average_processing_time_ms": 47.3,
    "plugin_version": "0.0.1a1",
    "configuration": {
        "confidence_threshold": 0.8,
        "model": "speechbrain/spkrec-ecapa-voxceleb"
    },
    "uptime_seconds": 3600.5,
    "enrolled_users": 5
}
```

#### Reset Statistics
```python
# Request
bus.emit(Message("ovos.voiceid.reset_stats", {}))

# Response: ovos.voiceid.stats.reset
{
    "status": "success"
}
```

### User Management

#### Enroll User
```python
# Audio data must be hex-encoded PCM audio
with open("user_audio.wav", "rb") as f:
    audio_data = f.read()
    audio_hex = audio_data.hex()

# Request
bus.emit(Message("ovos.voiceid.enroll_user", {
    "user_id": "john_doe",
    "audio_samples": [audio_hex]  # List of hex-encoded audio samples
}))

# Response: ovos.voiceid.enroll.response
{
    "user_id": "john_doe",
    "status": "success",
    "message": "User john_doe enrolled successfully with 1 audio samples",
    "samples_processed": 1
}
```

#### List Enrolled Users
```python
# Request
bus.emit(Message("ovos.voiceid.list_users", {}))

# Response: ovos.voiceid.list.response
{
    "users": ["john_doe", "jane_smith", "admin"],
    "count": 3,
    "status": "success"
}
```

#### Remove User
```python
# Request
bus.emit(Message("ovos.voiceid.remove_user", {
    "user_id": "john_doe"
}))

# Response: ovos.voiceid.remove.response
{
    "user_id": "john_doe",
    "status": "success",
    "message": "User john_doe removed successfully"
}
```

#### Update User Profile
```python
# Request - Add new samples or replace existing
bus.emit(Message("ovos.voiceid.update_user", {
    "user_id": "john_doe",
    "audio_samples": [audio_hex],
    "mode": "append"  # or "replace"
}))

# Response: ovos.voiceid.update.response
{
    "user_id": "john_doe",
    "status": "success",
    "message": "User john_doe updated successfully",
    "total_samples": 3
}
```

#### Get User Information
```python
# Request
bus.emit(Message("ovos.voiceid.get_user_info", {
    "user_id": "john_doe"
}))

# Response: ovos.voiceid.user.info.response
{
    "user_id": "john_doe",
    "status": "success",
    "sample_count": 3,
    "enrollment_date": "2025-08-30T12:00:00Z",
    "last_updated": "2025-08-30T14:30:00Z"
}
```

### Speaker Verification

#### Verify Two Audio Samples
```python
# Request
bus.emit(Message("ovos.voiceid.verify_speakers", {
    "audio_sample1": audio_hex_1,  # Direct hex string
    "audio_sample2": audio_hex_2   # Direct hex string
}))

# Response: ovos.voiceid.verify.response
{
    "is_same_speaker": true,
    "similarity_score": 0.8743,
    "status": "success",
    "confidence_threshold": 0.8,
    "message": "Verification complete: Same speaker (score: 0.874)"
}
```

## Event System

The plugin automatically emits identification events during audio processing.

### Voice Identification Events

#### Successful Identification
```python
# Event: ovos.voice.identified
{
    "speaker_id": "john_doe",
    "confidence": 0.8743,
    "processing_time_ms": 47.3,
    "plugin_version": "0.0.1a1",
    "timestamp": 1693401600.0
}
```

#### Unknown Voice Detection
```python
# Event: ovos.voice.unknown
{
    "confidence": 0.3421,
    "speaker_candidates": [],
    "fallback_mode": "guest",
    "plugin_version": "0.0.1a1",
    "timestamp": 1693401600.0
}
```

## Integration Examples

### OVOS Skill Integration

#### Basic Voice-Aware Skill
```python
from ovos_workshop.skills import OVOSSkill
from ovos_bus_client.message import Message

class VoiceAwareSkill(OVOSSkill):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_speaker = None
        self.speaker_confidence = 0.0

    def initialize(self):
        # Listen for voice identification events
        self.bus.on("ovos.voice.identified", self.handle_voice_identified)
        self.bus.on("ovos.voice.unknown", self.handle_voice_unknown)

    def handle_voice_identified(self, message):
        """Handle identified speaker events"""
        data = message.data
        self.current_speaker = data.get("speaker_id")
        self.speaker_confidence = data.get("confidence", 0.0)
        
        self.log.info(f"Speaker identified: {self.current_speaker} "
                     f"(confidence: {self.speaker_confidence:.3f})")
        
        # Customize response based on speaker
        if self.current_speaker == "admin":
            self.enable_admin_features()
        elif self.current_speaker == "child":
            self.enable_parental_controls()

    def handle_voice_unknown(self, message):
        """Handle unknown speaker events"""
        self.current_speaker = None
        self.speaker_confidence = message.data.get("confidence", 0.0)
        self.log.info("Unknown speaker detected")
        self.enable_guest_mode()

    @intent_handler("tell.me.who.am.i.intent")
    def handle_who_am_i(self, message):
        """Intent handler that uses voice identification"""
        if self.current_speaker:
            self.speak(f"You are {self.current_speaker}")
        else:
            self.speak("I don't recognize your voice")
```

#### User Enrollment Skill
```python
class VoiceEnrollmentSkill(OVOSSkill):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.enrollment_responses = {}

    def initialize(self):
        self.bus.on("ovos.voiceid.enroll.response", self.handle_enrollment_response)

    @intent_handler("enroll.my.voice.intent")
    def handle_enroll_voice(self, message):
        """Enroll user's voice"""
        user_id = message.data.get("user_id", "unknown_user")
        
        # Request audio recording
        self.speak("Please say something for 3 seconds to enroll your voice")
        
        # Record audio (implementation depends on your audio system)
        audio_data = self.record_audio(duration=3.0)
        audio_hex = audio_data.hex()
        
        # Send enrollment request
        self.bus.emit(Message("ovos.voiceid.enroll_user", {
            "user_id": user_id,
            "audio_samples": [audio_hex]
        }))

    def handle_enrollment_response(self, message):
        """Handle enrollment response"""
        data = message.data
        if data.get("status") == "success":
            user_id = data.get("user_id")
            self.speak(f"Voice enrollment successful for {user_id}")
        else:
            error_msg = data.get("message", "Unknown error")
            self.speak(f"Enrollment failed: {error_msg}")
```

### Plugin Integration

#### Audio Processing Pipeline Integration
```python
from ovos_plugin_manager.templates.audio_transformers import AudioTransformer

class CustomAudioProcessor(AudioTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.current_speaker = None
        self.bus.on("ovos.voice.identified", self.on_speaker_identified)

    def on_speaker_identified(self, message):
        """Adjust processing based on identified speaker"""
        speaker_id = message.data.get("speaker_id")
        self.current_speaker = speaker_id
        
        # Apply speaker-specific audio processing
        if speaker_id == "elderly_user":
            self.enable_noise_reduction()
        elif speaker_id == "child":
            self.adjust_volume_normalization()

    def transform(self, audio_data):
        """Process audio with speaker context"""
        if self.current_speaker:
            return self.apply_speaker_specific_processing(audio_data)
        return audio_data
```

### Mycroft Skills Integration
```python
from mycroft import MycroftSkill, intent_handler
from mycroft.messagebus.message import Message

class PersonalizedAssistantSkill(MycroftSkill):
    def __init__(self):
        super().__init__()
        self.user_preferences = {}
        self.current_user = None

    def initialize(self):
        self.add_event("ovos.voice.identified", self.handle_user_identified)
        self.add_event("ovos.voice.unknown", self.handle_unknown_user)
        
        # Load user preferences
        self.load_user_preferences()

    def handle_user_identified(self, message):
        """Personalize experience for identified user"""
        user_id = message.data.get("speaker_id")
        confidence = message.data.get("confidence", 0.0)
        
        if confidence >= 0.8:  # High confidence threshold
            self.current_user = user_id
            prefs = self.user_preferences.get(user_id, {})
            
            # Apply user preferences
            self.apply_user_settings(prefs)
            
            # Greet user
            greeting = prefs.get("greeting", f"Hello {user_id}")
            self.speak(greeting)

    @intent_handler("what.is.my.schedule.intent")
    def handle_schedule_request(self, message):
        """Provide personalized schedule"""
        if self.current_user:
            schedule = self.get_user_schedule(self.current_user)
            self.speak(f"Here's your schedule, {self.current_user}: {schedule}")
        else:
            self.speak("I need to identify you first. Please say something.")
```

## Audio Processing

### Audio Format Requirements

The plugin expects audio data in specific formats for different operations:

#### Input Audio Format
- **Sample Rate**: 16000 Hz (configurable)
- **Channels**: Mono preferred, stereo supported
- **Encoding**: PCM WAV format
- **Duration**: Minimum 1 second, optimal 2-5 seconds

#### Hex Encoding for Message Bus
```python
import wave

def prepare_audio_for_voiceid(audio_file_path, max_duration=3.0):
    """Prepare audio file for voice ID API"""
    with wave.open(audio_file_path, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        
        # Calculate frames to read
        max_frames = int(max_duration * sample_rate)
        frames_to_read = min(n_frames, max_frames)
        
        # Read and encode
        audio_data = wav_file.readframes(frames_to_read)
        return audio_data.hex()
```

### Processing Pipeline

1. **Audio Ingestion**: AudioTransformer receives audio chunks
2. **Preprocessing**: Audio normalization and resampling
3. **Feature Extraction**: ECAPA-TDNN embedding generation
4. **Identification**: Similarity comparison with enrolled users
5. **Event Emission**: Results broadcast via message bus

## Error Handling

### Common Error Scenarios

#### Enrollment Errors
```python
def handle_enrollment_error(self, message):
    """Handle enrollment error responses"""
    data = message.data
    error_msg = data.get("message", "Unknown error")
    
    if "User ID is required" in error_msg:
        self.log.error("Missing user ID in enrollment request")
    elif "Audio samples are required" in error_msg:
        self.log.error("No audio data provided for enrollment")
    elif "Voice processor not initialized" in error_msg:
        self.log.error("Plugin not properly initialized")
        # Attempt to reinitialize or alert admin
```

#### Verification Errors
```python
def handle_verification_error(self, message):
    """Handle verification error responses"""
    data = message.data
    if data.get("status") == "error":
        error_msg = data.get("message", "Unknown error")
        
        if "Two audio samples are required" in error_msg:
            self.log.error("Incomplete audio samples for verification")
        elif "Voice processor not initialized" in error_msg:
            self.log.error("Plugin initialization failure")
```

### Timeout Handling
```python
class VoiceIDClient:
    def __init__(self, bus):
        self.bus = bus
        self.pending_requests = {}

    def enroll_user_with_timeout(self, user_id, audio_samples, timeout=10.0):
        """Enroll user with timeout handling"""
        request_id = str(uuid.uuid4())
        
        def handle_response(message):
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]
                # Process response
                
        def handle_timeout():
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]
                self.log.error(f"Enrollment timeout for user {user_id}")
        
        # Set timeout
        timer = threading.Timer(timeout, handle_timeout)
        self.pending_requests[request_id] = timer
        timer.start()
        
        # Send request
        self.bus.emit(Message("ovos.voiceid.enroll_user", {
            "user_id": user_id,
            "audio_samples": audio_samples,
            "request_id": request_id
        }))
```

## Performance Considerations

### Resource Usage

#### Memory Optimization
- Model caching reduces initialization time
- User database is loaded into memory for fast access
- Audio buffers are released after processing

#### CPU/GPU Usage
```json
{
    "gpu": true,  // Enable for faster processing
    "processing_timeout_ms": 100,  // Prevent blocking
    "confidence_threshold": 0.8  // Balance accuracy vs performance
}
```

### Scaling Considerations

#### High-Volume Environments
- Monitor processing statistics via `ovos.voiceid.get_stats`
- Implement request queuing for enrollment operations
- Consider distributed deployment for multiple users

#### Optimization Tips
```python
# Monitor performance
def monitor_voiceid_performance(self):
    """Monitor Voice ID plugin performance"""
    self.bus.emit(Message("ovos.voiceid.get_stats", {}))
    
def handle_stats_response(self, message):
    """Analyze performance metrics"""
    stats = message.data
    avg_time = stats.get("average_processing_time_ms", 0)
    
    if avg_time > 100:  # Threshold for concern
        self.log.warning(f"Voice ID processing slow: {avg_time}ms")
        # Consider optimization or scaling
```

## Container Integration

### Docker Environment

The plugin is designed to work seamlessly in containerized OVOS environments:

```yaml
# docker-compose.yml excerpt
services:
  ovos_plugin_omva_voiceid:
    image: aczire/ovos_plugin_omva_voiceid:latest
    environment:
      - OVOS_CONFIG_BASE_FOLDER=/opt/omva/config
    volumes:
      - ./config:/opt/omva/config
      - ./models:/opt/omva/speechbrain_cache
    networks:
      - ovos_network
```

### Container Communication
```python
# Example container health check
def check_voiceid_health(self):
    """Check if Voice ID plugin is responsive"""
    response_received = False
    
    def handle_stats(message):
        nonlocal response_received
        response_received = True
        
    self.bus.on("ovos.voiceid.stats.response", handle_stats)
    self.bus.emit(Message("ovos.voiceid.get_stats", {}))
    
    # Wait for response
    time.sleep(2)
    return response_received
```

## Testing and Validation

### Unit Testing
```python
import unittest
from unittest.mock import Mock, patch
from ovos_bus_client.message import Message

class TestVoiceIDIntegration(unittest.TestCase):
    def setUp(self):
        self.bus = Mock()
        self.skill = VoiceAwareSkill()
        self.skill.bus = self.bus

    def test_voice_identification_handling(self):
        """Test voice identification event handling"""
        message = Message("ovos.voice.identified", {
            "speaker_id": "test_user",
            "confidence": 0.85,
            "timestamp": time.time()
        })
        
        self.skill.handle_voice_identified(message)
        
        self.assertEqual(self.skill.current_speaker, "test_user")
        self.assertEqual(self.skill.speaker_confidence, 0.85)
```

### Integration Testing
```python
def test_full_enrollment_cycle(self):
    """Test complete enrollment and identification cycle"""
    # Prepare test audio
    test_audio = self.load_test_audio("test_speaker.wav")
    audio_hex = test_audio.hex()
    
    # Test enrollment
    enrollment_response = self.send_message_and_wait(
        "ovos.voiceid.enroll_user",
        {"user_id": "test_user", "audio_samples": [audio_hex]},
        "ovos.voiceid.enroll.response"
    )
    
    self.assertEqual(enrollment_response.data["status"], "success")
    
    # Test verification
    verification_response = self.send_message_and_wait(
        "ovos.voiceid.verify_speakers",
        {"audio_sample1": audio_hex, "audio_sample2": audio_hex},
        "ovos.voiceid.verify.response"
    )
    
    self.assertTrue(verification_response.data["is_same_speaker"])
```

## Best Practices

### Security Considerations
1. **Audio Data Encryption**: Encrypt audio samples in transit
2. **User Privacy**: Implement data retention policies
3. **Access Control**: Restrict enrollment operations to authorized users

### Performance Best Practices
1. **Batch Processing**: Group enrollment operations when possible
2. **Caching**: Cache frequently accessed user embeddings
3. **Monitoring**: Implement comprehensive logging and monitoring

### Integration Guidelines
1. **Event-Driven Design**: Use message bus events for loose coupling
2. **Error Resilience**: Implement comprehensive error handling
3. **Configuration Management**: Use centralized configuration

## Conclusion

The OMVA Voice ID Plugin provides a robust foundation for building voice-aware applications in the OVOS ecosystem. Its comprehensive message bus API, real-time event system, and container-friendly design make it suitable for a wide range of integration scenarios, from simple skill personalization to complex multi-user systems.

For additional support and examples, refer to the plugin's test suite and the official OVOS documentation.
