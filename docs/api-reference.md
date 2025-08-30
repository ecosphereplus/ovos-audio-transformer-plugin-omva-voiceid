# OMVA Voice ID Plugin - API Reference

## Message Bus Events

### Requests (Outbound)

| Event | Purpose | Parameters |
|-------|---------|------------|
| `ovos.voiceid.get_stats` | Get plugin statistics | `{}` |
| `ovos.voiceid.reset_stats` | Reset plugin statistics | `{}` |
| `ovos.voiceid.enroll_user` | Enroll new user | `{"user_id": str, "audio_samples": [hex_str]}` |
| `ovos.voiceid.list_users` | List enrolled users | `{}` |
| `ovos.voiceid.remove_user` | Remove enrolled user | `{"user_id": str}` |
| `ovos.voiceid.update_user` | Update user profile | `{"user_id": str, "audio_samples": [hex_str], "mode": "append|replace"}` |
| `ovos.voiceid.get_user_info` | Get user information | `{"user_id": str}` |
| `ovos.voiceid.verify_speakers` | Verify if two samples are same speaker | `{"audio_sample1": hex_str, "audio_sample2": hex_str}` |

### Responses (Inbound)

| Event | Triggered By | Response Data |
|-------|-------------|---------------|
| `ovos.voiceid.stats.response` | `get_stats` | `{"total_processed": int, "successful_identifications": int, "failed_identifications": int, "average_processing_time_ms": float, "plugin_version": str, "configuration": {}, "uptime_seconds": float, "enrolled_users": int}` |
| `ovos.voiceid.stats.reset` | `reset_stats` | `{"status": "success"}` |
| `ovos.voiceid.enroll.response` | `enroll_user` | `{"user_id": str, "status": "success|error", "message": str, "samples_processed": int}` |
| `ovos.voiceid.list.response` | `list_users` | `{"users": [str], "count": int, "status": "success"}` |
| `ovos.voiceid.remove.response` | `remove_user` | `{"user_id": str, "status": "success|error", "message": str}` |
| `ovos.voiceid.update.response` | `update_user` | `{"user_id": str, "status": "success|error", "message": str, "total_samples": int}` |
| `ovos.voiceid.user.info.response` | `get_user_info` | `{"user_id": str, "status": "success|error", "sample_count": int, "enrollment_date": str, "last_updated": str}` |
| `ovos.voiceid.verify.response` | `verify_speakers` | `{"is_same_speaker": bool, "similarity_score": float, "status": "success|error", "confidence_threshold": float, "message": str}` |

### Automatic Events (Inbound)

| Event | When Emitted | Event Data |
|-------|-------------|------------|
| `ovos.voice.identified` | Speaker successfully identified | `{"speaker_id": str, "confidence": float, "processing_time_ms": float, "plugin_version": str, "timestamp": float}` |
| `ovos.voice.unknown` | Speaker not recognized or low confidence | `{"confidence": float, "speaker_candidates": [], "fallback_mode": "guest", "plugin_version": str, "timestamp": float}` |

## Code Examples

### Basic Event Listening
```python
from ovos_bus_client.message import Message

# Listen for voice identification
bus.on("ovos.voice.identified", lambda msg: print(f"User: {msg.data['speaker_id']}"))
bus.on("ovos.voice.unknown", lambda msg: print("Unknown speaker"))

# Get plugin stats
bus.emit(Message("ovos.voiceid.get_stats", {}))
bus.on("ovos.voiceid.stats.response", lambda msg: print(msg.data))
```

### User Management
```python
# Enroll user
bus.emit(Message("ovos.voiceid.enroll_user", {
    "user_id": "john_doe",
    "audio_samples": [audio_data.hex()]
}))

# List users
bus.emit(Message("ovos.voiceid.list_users", {}))

# Remove user
bus.emit(Message("ovos.voiceid.remove_user", {"user_id": "john_doe"}))
```

### Speaker Verification
```python
# Verify if two audio samples are from same speaker
bus.emit(Message("ovos.voiceid.verify_speakers", {
    "audio_sample1": sample1.hex(),
    "audio_sample2": sample2.hex()
}))

def handle_verification(message):
    result = message.data
    if result["is_same_speaker"]:
        print(f"Same speaker (score: {result['similarity_score']:.3f})")
    else:
        print(f"Different speakers (score: {result['similarity_score']:.3f})")

bus.on("ovos.voiceid.verify.response", handle_verification)
```

## Audio Format

### Input Requirements
- **Format**: PCM WAV
- **Sample Rate**: 16000 Hz (recommended)
- **Channels**: Mono preferred, stereo supported  
- **Duration**: 1-5 seconds optimal
- **Encoding**: Raw PCM bytes converted to hex string

### Audio Preparation
```python
import wave

def prepare_audio(file_path, max_duration=3.0):
    """Convert audio file to Voice ID format"""
    with wave.open(file_path, 'rb') as wav:
        rate = wav.getframerate()
        frames = wav.readframes(int(max_duration * rate))
        return frames.hex()

# Usage
audio_hex = prepare_audio("user_sample.wav")
```

## Error Codes

### Common Error Messages

| Message | Cause | Solution |
|---------|-------|----------|
| `"User ID is required for enrollment"` | Missing `user_id` parameter | Provide valid `user_id` string |
| `"Audio samples are required for enrollment"` | Missing or empty `audio_samples` | Provide hex-encoded audio data |
| `"Voice processor not initialized"` | Plugin initialization failed | Check plugin configuration and logs |
| `"Two audio samples are required for verification"` | Missing audio samples in verification | Provide both `audio_sample1` and `audio_sample2` |
| `"User [id] not found"` | User not enrolled | Enroll user first before operations |

### Error Handling Pattern
```python
def handle_response(message):
    data = message.data
    if data.get("status") == "error":
        error_msg = data.get("message", "Unknown error")
        print(f"Operation failed: {error_msg}")
        return False
    return True
```

## Configuration Reference

```json
{
    "listener": {
        "audio_transformers": {
            "ovos-audio-transformer-plugin-omva-voiceid": {
                "enabled": true,
                "model": "speechbrain/spkrec-ecapa-voxceleb",
                "confidence_threshold": 0.8,
                "sample_rate": 16000,
                "gpu": false,
                "enable_enrollment": true,
                "processing_timeout_ms": 100,
                "model_cache_dir": "/tmp/omva_models"
            }
        }
    }
}
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Enable/disable plugin |
| `model` | str | `"speechbrain/spkrec-ecapa-voxceleb"` | SpeechBrain model identifier |
| `confidence_threshold` | float | `0.8` | Minimum confidence for positive identification |
| `sample_rate` | int | `16000` | Expected audio sample rate |
| `gpu` | bool | `false` | Enable GPU acceleration |
| `enable_enrollment` | bool | `true` | Allow user enrollment operations |
| `processing_timeout_ms` | int | `100` | Maximum processing time per audio chunk |
| `model_cache_dir` | str | `"/tmp/omva_models"` | Directory for model cache |

## Performance Metrics

### Statistics Response
```json
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

### Performance Monitoring
```python
def monitor_performance():
    def handle_stats(message):
        stats = message.data
        avg_time = stats.get("average_processing_time_ms", 0)
        success_rate = stats.get("successful_identifications", 0) / max(stats.get("total_processed", 1), 1)
        
        print(f"Avg processing time: {avg_time:.1f}ms")
        print(f"Success rate: {success_rate:.1%}")
        
        if avg_time > 100:
            print("WARNING: High processing time detected")
    
    bus.on("ovos.voiceid.stats.response", handle_stats)
    bus.emit(Message("ovos.voiceid.get_stats", {}))
```

## Container Integration

### Docker Environment Variables
```bash
OVOS_CONFIG_BASE_FOLDER=/opt/omva/config
VOICEID_MODEL_CACHE=/opt/omva/cache
VOICEID_GPU_ENABLED=false
```

### Health Check
```python
def check_voiceid_health():
    """Verify plugin is responsive"""
    response_received = False
    
    def handle_stats(message):
        nonlocal response_received
        response_received = True
        version = message.data.get("plugin_version")
        print(f"Voice ID plugin healthy, version: {version}")
    
    bus.on("ovos.voiceid.stats.response", handle_stats)
    bus.emit(Message("ovos.voiceid.get_stats", {}))
    
    time.sleep(2)
    return response_received
```

## Testing Utilities

### Message Bus Test Helper
```python
class VoiceIDTester:
    def __init__(self, bus):
        self.bus = bus
        self.responses = {}
    
    def send_and_wait(self, request_event, data, response_event, timeout=5):
        """Send message and wait for response"""
        response = None
        
        def handler(message):
            nonlocal response
            response = message.data
        
        self.bus.on(response_event, handler)
        self.bus.emit(Message(request_event, data))
        
        start = time.time()
        while response is None and time.time() - start < timeout:
            time.sleep(0.1)
        
        return response

# Usage
tester = VoiceIDTester(bus)
stats = tester.send_and_wait(
    "ovos.voiceid.get_stats", {}, 
    "ovos.voiceid.stats.response"
)
```

## Migration Guide

### From Direct Plugin Usage
```python
# Old (direct plugin usage)
from ovos_audio_transformer_plugin_omva_voiceid import OMVAVoiceIDPlugin
plugin = OMVAVoiceIDPlugin({})
speaker_id, confidence = plugin.identify_speaker(audio_data)

# New (message bus integration)
bus.emit(Message("ovos.voiceid.verify_speakers", {
    "audio_sample1": known_sample.hex(),
    "audio_sample2": test_sample.hex()
}))
bus.on("ovos.voiceid.verify.response", handle_verification_result)
```

### Event Name Changes
| Old Event | New Event | Notes |
|-----------|-----------|-------|
| `voice.identified` | `ovos.voice.identified` | Added OVOS namespace |
| `voice.unknown` | `ovos.voice.unknown` | Added OVOS namespace |
| Custom events | `ovos.voiceid.*` | Standardized naming |
