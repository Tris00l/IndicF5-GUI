# IndicF5-GUI — Multilingual TTS on AMD GPU (ROCm)

Futuristic Gradio web app for **text-to-speech in 11 Indian languages** using [ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5), running on **AMD GPUs via ROCm** (no NVIDIA required).

## Languages

| Language | Code | Male Sample | Female Sample |
|----------|------|-------------|---------------|
| Bengali | `bn` | bn_male.wav | — |
| Gujarati | `gu` | gu_male.wav | gu_female.wav |
| Hindi | `hi` | hi_male.wav | hi_female.wav |
| Kannada | `kn` | kn_male.wav | kn_female.wav |
| Malayalam | `ml` | ml_male.wav | ml_female.wav |
| Marathi | `mr` | mr_male.wav | mr_female.wav |
| Tamil | `ta` | ta_male.wav | ta_female.wav |
| Telugu | `te` | te_male.wav | te_female.wav |

## Features

- Paste text in any supported Indian language → generate speech in seconds
- Pre-loaded voice presets for male & female voices per language (no reference audio upload needed)
- Futuristic dark UI built with Gradio 6
- ROCm / HIP compatible (tested on RDNA3 / gfx1100)

## Project Layout

```
IndicF5-GUI/
├── indicf5_app.py         # Main Gradio app
├── models/
│   ├── model.safetensors  → symlink to HF cache (ai4bharat/IndicF5)
│   └── vocab.txt          → symlink to HF cache (Indic script vocabulary)
├── samples/               # Reference audio for voice presets
│   ├── hi_male.wav
│   ├── ta_female.wav
│   └── ...
├── README.md
├── SETUP.md
└── .gitignore
```

## Quick Start

```bash
# One-time setup — see SETUP.md for full details
conda activate indicf5
python indicf5_app.py
```

Open `http://localhost:7860` in your browser.

## Usage

1. Select a language and voice preset from the dropdown
2. Type or paste text in the text box
3. Click **Generate**
4. Listen to or download the generated audio

## Model

| Component | Source |
|-----------|--------|
| TTS model | [ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5) (gated — accept terms once) |
| Vocoder | [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz) |
| Vocabulary | IndicF5 `checkpoints/vocab.txt` (2545 tokens) |

## Citation

If you use this work, please cite the original model:

```bibtex
@misc{AI4Bharat_IndicF5_2025,
  author       = {Praveen S V and Srija Anand and Soma Siddhartha and Mitesh M. Khapra},
  title        = {IndicF5: High-Quality Text-to-Speech for Indian Languages},
  year         = {2025},
  url          = {https://github.com/AI4Bharat/IndicF5},
}
```

## Hardware

- **GPU**: AMD RX 7900 XTX (gfx1100 / RDNA3)
- **ROCm**: 7.2 / HIP 7.2
- **PyTorch**: 2.9.1+rocm6.4

## Known Workarounds Applied

- `torchaudio.load()` patched to use `soundfile` backend — torchcodec incompatible with ROCm torch
- `f5_tts.infer.utils_infer.load_model` monkey-patched — package ships broken 7-arg signature
- Checkpoint loading strips `_orig_mod.` prefix (artifact of `torch.compile()`) and `vocoder.*` keys
