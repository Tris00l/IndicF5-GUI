# IndicF5-GUI — Multilingual TTS for Indian Languages

Futuristic Gradio web app for **text-to-speech in 11 Indian languages** using [ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5), supporting **AMD GPUs via ROCm** and **NVIDIA GPUs via CUDA**.

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
- ROCm / CUDA compatible (tested on RDNA3 and NVIDIA GPUs)

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
├── SETUP.md                   # ROCm (AMD) setup
├── requirements-nvidia.txt    # CUDA (NVIDIA) setup
└── .gitignore
```

## Quick Start

**AMD GPU (ROCm):** see [SETUP.md](SETUP.md) for the full setup guide.

**NVIDIA GPU (CUDA):**
```bash
conda create -n indicf5 python=3.10 -y
conda activate indicf5
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/ai4bharat/IndicF5.git
pip install -r requirements-nvidia.txt
# Download model (one-time)
huggingface-cli login
python -c "from huggingface_hub import hf_hub_download; \
  hf_hub_download('ai4bharat/IndicF5', 'model.safetensors'); \
  hf_hub_download('ai4bharat/IndicF5', 'checkpoints/vocab.txt')"
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

**AMD (ROCm)**
- **GPU**: AMD RX 7900 XTX (gfx1100 / RDNA3)
- **ROCm**: 7.2 / HIP 7.2
- **PyTorch**: 2.5.1+rocm6.4

**NVIDIA (CUDA)**
- **GPU**: any CUDA-compatible NVIDIA GPU
- **CUDA**: 12.4
- **PyTorch**: 2.5.1+cu124

## Known Workarounds Applied

- `torchaudio.load()` patched to use `soundfile` backend — needed for ROCm (torchcodec incompatible), harmless on CUDA
- `f5_tts.infer.utils_infer.load_model` monkey-patched — package ships broken 7-arg signature
- Checkpoint loading strips `_orig_mod.` prefix (artifact of `torch.compile()`) and `vocoder.*` keys
