# OMVA Voice ID Plugin - Quick Start Integration Guide

## Overview

This guide provides quick integration examples for common use cases with the OMVA Voice ID Plugin in OVOS environments.

## Quick Setup

### 1. Install and Configure
```bash
pip install ovos-audio-transformer-plugin-omva-voiceid
```

Add to your OVOS configuration:
```json
{
    "listener": {
        "audio_transformers": {
            "ovos-audio-transformer-plugin-omva-voiceid": {
                "enabled": true,
                "confidence_threshold": 0.8
            }
        }
    }
}
```

### 2. Basic Integration Examples

#### Simple Voice-Aware Skill
```python
from ovos_workshop.skills import OVOSSkill

class MyVoiceSkill(OVOSSkill):
    def initialize(self):
        self.current_user = None
        self.bus.on("ovos.voice.identified", self.on_voice_identified)
        self.bus.on("ovos.voice.unknown", self.on_voice_unknown)

    def on_voice_identified(self, message):
        self.current_user = message.data.get("speaker_id")
        confidence = message.data.get("confidence", 0)
        self.log.info(f"User identified: {self.current_user} ({confidence:.2%})")

    def on_voice_unknown(self, message):
        self.current_user = None
        self.log.info("Unknown user detected")

    @intent_handler("who.am.i.intent")
    def handle_who_am_i(self, message):
        if self.current_user:
            self.speak(f"You are {self.current_user}")
        else:
            self.speak("I don't recognize you")
```

#### User Enrollment
```python
@intent_handler("enroll.my.voice.intent")
def handle_voice_enrollment(self, message):
    user_name = self.get_response("What's your name?")
    if not user_name:
        return
    
    self.speak("Please speak for 3 seconds")
    audio_data = self.record_audio(duration=3.0)
    
    # Send enrollment request
    self.bus.emit(Message("ovos.voiceid.enroll_user", {
        "user_id": user_name.lower().replace(" ", "_"),
        "audio_samples": [audio_data.hex()]
    }))
    
    # Wait for response
    self.bus.on("ovos.voiceid.enroll.response", self.handle_enrollment_result)

def handle_enrollment_result(self, message):
    if message.data.get("status") == "success":
        user_id = message.data.get("user_id")
        self.speak(f"Successfully enrolled {user_id}")
    else:
        self.speak("Enrollment failed. Please try again.")
```

#### Personalized Responses
```python
class PersonalizedSkill(OVOSSkill):
    def initialize(self):
        self.user_preferences = {
            "alice": {"greeting": "Hello Alice!", "voice": "female"},
            "bob": {"greeting": "Hey there Bob!", "voice": "male"}
        }
        self.current_user = None
        self.bus.on("ovos.voice.identified", self.on_user_identified)

    def on_user_identified(self, message):
        self.current_user = message.data.get("speaker_id")
        prefs = self.user_preferences.get(self.current_user, {})
        
        # Apply user preferences
        if "voice" in prefs:
            self.settings["voice"] = prefs["voice"]

    @intent_handler("good.morning.intent")
    def handle_good_morning(self, message):
        if self.current_user in self.user_preferences:
            greeting = self.user_preferences[self.current_user]["greeting"]
            self.speak(greeting)
        else:
            self.speak("Good morning!")
```

### 3. Message Bus API Quick Reference

```python
from ovos_bus_client.message import Message

# Get plugin statistics
bus.emit(Message("ovos.voiceid.get_stats", {}))

# Enroll user (audio_hex is hex-encoded PCM audio)
bus.emit(Message("ovos.voiceid.enroll_user", {
    "user_id": "john_doe",
    "audio_samples": [audio_hex]
}))

# List enrolled users
bus.emit(Message("ovos.voiceid.list_users", {}))

# Verify two audio samples are from same speaker
bus.emit(Message("ovos.voiceid.verify_speakers", {
    "audio_sample1": audio_hex_1,
    "audio_sample2": audio_hex_2
}))

# Remove user
bus.emit(Message("ovos.voiceid.remove_user", {
    "user_id": "john_doe"
}))
```

### 4. Common Events to Listen For

```python
def initialize(self):
    # Core identification events
    self.bus.on("ovos.voice.identified", self.handle_identified)
    self.bus.on("ovos.voice.unknown", self.handle_unknown)
    
    # Management response events
    self.bus.on("ovos.voiceid.stats.response", self.handle_stats)
    self.bus.on("ovos.voiceid.enroll.response", self.handle_enrollment)
    self.bus.on("ovos.voiceid.list.response", self.handle_user_list)
    self.bus.on("ovos.voiceid.verify.response", self.handle_verification)
```

### 5. Audio Processing Helper

```python
import wave

def prepare_audio_sample(file_path, duration=2.0):
    """Convert audio file to format suitable for Voice ID plugin"""
    with wave.open(file_path, 'rb') as wav:
        frames = wav.readframes(int(duration * wav.getframerate()))
        return frames.hex()

# Usage
audio_hex = prepare_audio_sample("user_voice.wav", duration=3.0)
```

### 6. Container Integration

```python
# Health check for containerized environments
def check_voiceid_status(self):
    """Check if Voice ID plugin is running"""
    self.stats_received = False
    
    def handle_stats(message):
        self.stats_received = True
        self.voiceid_version = message.data.get("plugin_version")
    
    self.bus.on("ovos.voiceid.stats.response", handle_stats)
    self.bus.emit(Message("ovos.voiceid.get_stats", {}))
    
    # Wait briefly for response
    time.sleep(1)
    return self.stats_received
```

### 7. Error Handling Pattern

```python
def safe_voice_operation(self, operation_type, data, callback):
    """Generic error-safe voice operation"""
    response_event = f"ovos.voiceid.{operation_type}.response"
    timeout = 5.0
    
    def handle_response(message):
        if message.data.get("status") == "success":
            callback(True, message.data)
        else:
            error_msg = message.data.get("message", "Unknown error")
            self.log.error(f"Voice ID {operation_type} failed: {error_msg}")
            callback(False, message.data)
    
    def handle_timeout():
        self.log.error(f"Voice ID {operation_type} timeout")
        callback(False, {"message": "Operation timed out"})
    
    # Set up handlers
    self.bus.on(response_event, handle_response)
    timer = threading.Timer(timeout, handle_timeout)
    timer.start()
    
    # Send request
    request_event = f"ovos.voiceid.{operation_type}"
    self.bus.emit(Message(request_event, data))

# Usage
def enroll_user_safely(self, user_id, audio_samples):
    def enrollment_callback(success, data):
        if success:
            self.speak(f"Successfully enrolled {user_id}")
        else:
            self.speak("Enrollment failed")
    
    self.safe_voice_operation("enroll_user", {
        "user_id": user_id,
        "audio_samples": audio_samples
    }, enrollment_callback)
```

### 8. Testing Your Integration

```python
# Simple test to verify integration
class TestVoiceIntegration:
    def test_plugin_connectivity(self):
        """Test if plugin responds to stats request"""
        response = None
        
        def handle_stats(message):
            nonlocal response
            response = message.data
        
        bus.on("ovos.voiceid.stats.response", handle_stats)
        bus.emit(Message("ovos.voiceid.get_stats", {}))
        
        time.sleep(2)
        assert response is not None
        assert "plugin_version" in response
        print(f"✅ Plugin connected, version: {response['plugin_version']}")
```

### 9. Configuration Tips

```json
{
    "listener": {
        "audio_transformers": {
            "ovos-audio-transformer-plugin-omva-voiceid": {
                "enabled": true,
                "confidence_threshold": 0.8,
                "processing_timeout_ms": 100,
                "enable_enrollment": true,
                "gpu": false,
                "model_cache_dir": "/opt/omva/cache"
            }
        }
    }
}
```

### 10. Troubleshooting

```python
# Debug helper
def debug_voiceid_status(self):
    """Print Voice ID plugin status"""
    def handle_stats(message):
        data = message.data
        print(f"Voice ID Status:")
        print(f"  Version: {data.get('plugin_version')}")
        print(f"  Processed: {data.get('total_processed', 0)}")
        print(f"  Users: {data.get('enrolled_users', 0)}")
        print(f"  Avg Time: {data.get('average_processing_time_ms', 0):.1f}ms")
    
    self.bus.on("ovos.voiceid.stats.response", handle_stats)
    self.bus.emit(Message("ovos.voiceid.get_stats", {}))
```

## Next Steps

- Review the [Technical Integration Guide](technical-integration-guide.md) for detailed API documentation
- Check the test files for more comprehensive examples
- Join the OVOS community for support and discussions

## Support

For issues and questions:
- Check the plugin logs: `/var/log/ovos/` 
- Monitor message bus traffic
- Verify audio format compatibility (16kHz PCM preferred)
