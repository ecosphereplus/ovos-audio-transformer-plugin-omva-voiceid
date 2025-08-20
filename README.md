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
                  "engine": "speechbrain",
                  "model": "speechbrain/spkrec-ecapa-voxceleb",
                  "verification_threshold": 0.25,
                  "model_cache_dir": "/opt/omva/speechbrain_cache",
                  "gpu": true,
                  "confidence_threshold": 0.8,
                  "processing_timeout_ms": 100,
                  "voice_processing": {
                      "sample_rate": 16000,
                      "mfcc_coefficients": 13
                  },
                  "enrollment": {
                      "min_samples": 3,
                      "max_samples": 5,
                      "sample_duration": 3.0
                  },
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
