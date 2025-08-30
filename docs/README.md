# OMVA Voice ID Plugin Documentation

Welcome to the OMVA Voice ID Plugin documentation. This plugin provides advanced speaker identification and verification capabilities for the OpenVoiceOS ecosystem.

## Documentation Overview

### 🚀 [Quick Start Integration Guide](quick-start-integration.md)
Perfect for developers who want to get up and running quickly. Contains:
- Simple setup instructions
- Basic integration examples  
- Common use cases
- Troubleshooting tips

### 📚 [Technical Integration Guide](technical-integration-guide.md)  
Comprehensive technical documentation for advanced integrations. Includes:
- Complete architecture overview
- Detailed message bus API
- Integration patterns and examples
- Performance optimization
- Container deployment
- Security considerations

### 📖 [API Reference](api-reference.md)
Concise reference for all API endpoints and events:
- Complete message bus event catalog
- Parameter specifications
- Response formats
- Error codes and handling
- Configuration options

## Key Features

- **Real-time Speaker Identification**: Continuous voice recognition with confidence scoring
- **Multi-user Support**: Enroll and manage multiple voice profiles
- **Speaker Verification**: Compare two audio samples for speaker matching
- **Message Bus Integration**: Complete OVOS ecosystem integration
- **Container Ready**: Docker and containerized deployment support
- **High Performance**: Optimized with SpeechBrain ECAPA-TDNN models

## Quick Examples

### Listen for Voice Events
```python
# Simple voice identification listener
bus.on("ovos.voice.identified", lambda msg: print(f"User: {msg.data['speaker_id']}"))
bus.on("ovos.voice.unknown", lambda msg: print("Unknown speaker"))
```

### Enroll a New User
```python
# Enroll user with audio sample
bus.emit(Message("ovos.voiceid.enroll_user", {
    "user_id": "alice",
    "audio_samples": [audio_data.hex()]
}))
```

### Get Plugin Statistics
```python
# Monitor plugin performance
bus.emit(Message("ovos.voiceid.get_stats", {}))
bus.on("ovos.voiceid.stats.response", lambda msg: print(msg.data))
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   OVOS Skills   │    │  Other Plugins   │    │  Applications   │
└─────────┬───────┘    └─────────┬────────┘    └─────────┬───────┘
          │                      │                       │
          └──────────────────────┼───────────────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │     OVOS Message Bus     │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  OMVA Voice ID Plugin    │
                    │  ┌─────────────────────┐ │
                    │  │ AudioTransformer    │ │
                    │  └─────────────────────┘ │
                    │  ┌─────────────────────┐ │
                    │  │ SpeechBrain Model   │ │
                    │  └─────────────────────┘ │
                    │  ┌─────────────────────┐ │
                    │  │ User Database       │ │
                    │  └─────────────────────┘ │
                    └─────────────────────────┘
```

## Integration Scenarios

### 🏠 Smart Home Personalization
- User-specific device preferences
- Personalized responses and greetings
- Security and access control

### 🎯 Skill Development
- Voice-aware OVOS skills
- User context in conversations
- Multi-user applications

### 🔒 Security Applications  
- Voice-based authentication
- Speaker verification systems
- Access control integration

### 🏢 Enterprise Integration
- Multi-tenant voice systems
- User analytics and insights
- Compliance and audit trails

## Message Bus Events

### Core Events
- `ovos.voice.identified` - Speaker successfully identified
- `ovos.voice.unknown` - Unknown or low-confidence speaker

### Management Events
- `ovos.voiceid.enroll_user` - Enroll new speaker
- `ovos.voiceid.verify_speakers` - Compare two audio samples
- `ovos.voiceid.get_stats` - Get plugin statistics
- `ovos.voiceid.list_users` - List enrolled users

## Installation & Setup

```bash
# Install the plugin
pip install ovos-audio-transformer-plugin-omva-voiceid

# Configure in OVOS
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

## Support & Community

- **GitHub Repository**: [ovos-audio-transformer-plugin-omva-voiceid](https://github.com/ecosphereplus/ovos-audio-transformer-plugin-omva-voiceid)
- **OVOS Community**: Join the OpenVoiceOS community for support and discussions
- **Issue Tracking**: Report bugs and request features on GitHub
- **Documentation**: This documentation is continuously updated

## License

This plugin is released under an open-source license. See the LICENSE file for details.

## Contributing

We welcome contributions! Please see the contributing guidelines in the main repository.

---

## Next Steps

1. **New to OVOS?** Start with the [Quick Start Guide](quick-start-integration.md)
2. **Building Skills?** Check the [Technical Integration Guide](technical-integration-guide.md)  
3. **Need API Details?** Reference the [API Documentation](api-reference.md)
4. **Having Issues?** See troubleshooting sections in each guide

Happy coding! 🎉
