# OMVA Voice Identification Plugin for OVOS

A multi-user voice identification plugin for the OpenVoiceOS ecosystem that provides real-time speaker identification with advanced confidence calibration.

## Features

- Real-time speaker identification with enhanced discrimination
- Advanced confidence calibration system (2.46x discrimination ratio)
- OVOS AudioTransformer integration
- Container and standalone service support
- Message bus event emission
- Multi-user enrollment system
- GPU acceleration support

## Installation

### Standard Installation

```bash
pip install -e .
```

### Standalone Service

The plugin can run as a standalone service:

```bash
# Using the console script
ovos-omva-voiceid-listener

# Or direct Python invocation  
python -m ovos_audio_transformer_plugin_omva_voiceid
```

## Configuration

Add to your OVOS configuration:

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
                "enable_enrollment": true,
                "enrollment": {
                    "min_samples": 3,
                    "max_samples": 5,
                    "sample_duration": 3.0
                }
            }
        }
    }
}
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `true` | Enable/disable the plugin |
| `model` | string | `"speechbrain/spkrec-ecapa-voxceleb"` | SpeechBrain model identifier |
| `model_cache_dir` | string | `"~/.local/share/omva_voiceid"` | Directory for model cache and user database |
| `sample_rate` | integer | `16000` | Audio sample rate in Hz |
| `confidence_threshold` | float | `0.8` | Minimum confidence for speaker identification |
| `processing_timeout_ms` | integer | `100` | Maximum processing time in milliseconds |
| `gpu` | boolean | `false` | Enable GPU acceleration (requires CUDA) |
| `enable_enrollment` | boolean | `true` | Enable user enrollment functionality |
| `enrollment.min_samples` | integer | `3` | Minimum audio samples for user enrollment |
| `enrollment.max_samples` | integer | `5` | Maximum audio samples for user enrollment |
| `enrollment.sample_duration` | float | `3.0` | Duration of each enrollment sample in seconds |

## Events

The plugin emits the following message bus events:

### `ovos.voice.identified`
Emitted when a speaker is successfully identified with confidence above the threshold.

**Data:**
```json
{
    "speaker_id": "user_123",
    "confidence": 0.85,
    "processing_time_ms": 45.2,
    "plugin_version": "0.0.1",
    "timestamp": 1692528000.123
}
```

### `ovos.voice.unknown` 
Emitted when no speaker is identified or confidence is below the threshold.

**Data:**
```json
{
    "confidence": 0.15,
    "speaker_candidates": [
        {"speaker_id": "user_123", "confidence": 0.15}
    ],
    "fallback_mode": "guest",
    "plugin_version": "0.0.1", 
    "timestamp": 1692528000.123
}
```

## Message Bus Control

The plugin responds to the following message bus commands:

- `ovos.voiceid.get_stats`: Get processing statistics
- `ovos.voiceid.reset_stats`: Reset processing statistics  
- `ovos.voiceid.enroll_user`: Enroll a new user with audio samples
- `ovos.voiceid.update_user`: Update existing user with new audio samples
- `ovos.voiceid.remove_user`: Remove enrolled user from database
- `ovos.voiceid.list_users`: List all enrolled users
- `ovos.voiceid.get_user_info`: Get detailed information about a specific user
- `ovos.voiceid.verify_speakers`: Verify if two audio samples are from the same speaker

## Testing

### Unit Tests

Run basic unit tests:

```bash
python -m pytest tests/test_plugin.py -v
```

### Integration Tests (Container-Based)

The plugin includes comprehensive integration tests designed to run inside the OVOS container environment to validate real-world deployment scenarios.

**Container Testing Overview:**
- Tests run inside the actual OVOS/OMVA container
- Validates message bus communication in containerized environment  
- Uses real Obama and JFK presidential speech samples
- Tests complete user lifecycle management via message bus only

**Required Audio Files:**
The integration tests require Obama and JFK presidential speech samples:

- `obama.wav` - Main Obama speech sample
- `jfk.wav` - Main JFK speech sample  
- `obama-val.wav` - Obama validation sample
- `jfk-val.wav` - JFK validation sample

**Quick Container Testing:**

```bash
docker cp obama*.wav <container>:/opt/omva/audio/
docker cp jfk*.wav <container>:/opt/omva/audio/
docker cp tests/test_integration.py <container>:/opt/omva/container_test.py
docker exec -it <container> python /opt/omva/container_test.py
```

**Test Coverage:**

The container integration tests validate:

- ✅ Container environment detection and validation
- ✅ OVOS message bus running and accessible  
- ✅ OMVA Voice ID plugin loaded and responsive
- ✅ Real audio file loading and processing
- ✅ Multi-user enrollment via message bus
- ✅ User profile updates with validation audio
- ✅ Speaker verification with real voices (Obama vs JFK)
- ✅ Complete user lifecycle management
- ✅ Message bus event handling and responses
- ✅ Container-specific configuration and paths

**Expected Results:**
With properly configured container environment, tests achieve 100% success rate demonstrating:
- Same speaker similarity: ~0.745 (Obama vs Obama-val)  
- Different speaker similarity: ~0.066 (Obama vs JFK)
- Discrimination gap: ~0.679 (excellent separation)
- All message bus endpoints functional
- Real-time audio processing working perfectly

**Troubleshooting Container Tests:**

If tests fail, check:
1. OVOS message bus is running: `docker exec <container> pgrep -f ovos-messagebus`
2. Plugin is loaded: `docker exec <container> pgrep -f omva`
3. Audio files are accessible: `docker exec <container> ls -la /opt/omva/audio/`
4. Python environment: `docker exec <container> python -c "import ovos_bus_client; print('OK')"`

## Development

This plugin is part of the OMVA project for Issues #63 and #64:
- Enhanced OVOS Dinkum Listener integration
- Real-time voice identification capability

## License

Apache 2.0
