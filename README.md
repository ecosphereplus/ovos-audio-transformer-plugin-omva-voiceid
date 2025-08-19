# OMVA Voice Identification Plugin for OVOS

A multi-user voice identification plugin for the OpenVoiceOS ecosystem.

## Features

- Real-time speaker identification
- OVOS AudioTransformer integration
- Message bus event emission
- Multi-user support

## Installation

```bash
pip install -e .
```

## Configuration

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

## Events

- `ovos.voice.identified`: Speaker identified successfully
- `ovos.voice.unknown`: Speaker not identified

## Development

This plugin is part of the OMVA project for Issues #63 and #64:
- Enhanced OVOS Dinkum Listener integration
- Real-time voice identification capability

## License

Apache 2.0
