# VoiRS Model Zoo

This directory contains pre-trained models for VoiRS speech synthesis.

## Directory Structure

```
models/
├── acoustic/           # Text-to-Mel models
│   ├── vits/          # VITS models
│   └── fastspeech2/   # FastSpeech2 models
├── vocoder/           # Mel-to-Wave models
│   ├── hifigan/       # HiFi-GAN vocoders
│   └── diffwave/      # DiffWave vocoders
└── g2p/              # Grapheme-to-Phoneme models
```

## Model Downloads

Models are downloaded automatically when first used, or can be pre-downloaded using:

```bash
voirs download --model all
voirs download --model vits_en_us
voirs download --model hifigan_22k
```

## Supported Models

### Acoustic Models (Text → Mel)
- `vits_en_us`: English VITS model trained on LJSpeech
- `vits_multilingual`: Multilingual VITS supporting 10+ languages
- `fastspeech2_en`: FastSpeech2 for English

### Vocoders (Mel → Wave)
- `hifigan_22k`: HiFi-GAN vocoder for 22kHz synthesis
- `hifigan_48k`: HiFi-GAN vocoder for 48kHz synthesis
- `diffwave_22k`: DiffWave vocoder for high-quality synthesis

### G2P Models
- `cmudict.fst`: CMU pronunciation dictionary (English)
- `g2p_multilingual`: Lightweight neural G2P for multiple languages

## Model Formats

- **Candle**: `.safetensors` format (preferred)
- **ONNX**: `.onnx` format for cross-platform compatibility
- **FST**: Finite State Transducers for G2P

## License Information

Each model directory contains a `LICENSE` file with licensing information.
Most models are released under permissive licenses (MIT/Apache-2.0/Creative Commons).