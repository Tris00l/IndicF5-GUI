# Setup Guide — IndicF5 on AMD GPU (ROCm)

Tested on: Arch Linux, ROCm 7.2, AMD RX 7900 XTX (gfx1100 / RDNA3)

---

## 1. Prerequisites

- AMD GPU with ROCm support
- ROCm 7.2+ installed (`/opt/rocm` exists)
- Miniconda or Anaconda

Verify ROCm:
```bash
rocminfo | grep "gfx"
```

---

## 2. Create Conda Environment

```bash
conda create -n indicf5 python=3.10 -y
conda activate indicf5
```

---

## 3. Install ROCm PyTorch

> Do NOT use the default `pip install torch` — that installs a CUDA build.

```bash
pip install torch==2.9.1 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/rocm6.4
```

Verify GPU is visible:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
# Expected: True  AMD Radeon RX 7900 XTX
```

Set environment variable for RDNA3 (gfx1100) if needed:
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
# Add to ~/.bashrc or conda activate script to persist
```

---

## 4. Install IndicF5 Package

```bash
pip install git+https://github.com/ai4bharat/IndicF5.git
```

> The package installs as `f5_tts`. It ships with a broken `load_model` signature
> and no `vocab.txt` — the app monkey-patches both at startup automatically.

---

## 5. Install Dependencies

```bash
pip install gradio>=6.0 soundfile huggingface_hub safetensors transformers
```

Remove torchcodec if present (incompatible with ROCm torch):
```bash
pip uninstall torchcodec -y 2>/dev/null || true
```

---

## 6. Authenticate with HuggingFace

The IndicF5 model is gated (free, requires one-time accept).

```bash
huggingface-cli login
# paste your HF token from https://huggingface.co/settings/tokens
```

Then visit **https://huggingface.co/ai4bharat/IndicF5** and click **"Agree and access repository"**.

---

## 7. Download Model Assets

Run once to pre-download (~1.4 GB):

```bash
conda activate indicf5
python - <<'EOF'
from huggingface_hub import hf_hub_download
ckpt  = hf_hub_download("ai4bharat/IndicF5", "model.safetensors")
vocab = hf_hub_download("ai4bharat/IndicF5", "checkpoints/vocab.txt")
print("model:", ckpt)
print("vocab:", vocab)
EOF
```

Then create symlinks into the project folder:

```bash
SNAP=$(python -c "
from huggingface_hub import hf_hub_download
import os
p = hf_hub_download('ai4bharat/IndicF5', 'model.safetensors')
print(os.path.dirname(p))
")

PROJ_DIR="$PWD"   # or wherever you cloned IndicF5-GUI
ln -sf "$SNAP/model.safetensors"      "$PROJ_DIR/models/model.safetensors"
ln -sf "$SNAP/../checkpoints/vocab.txt" "$PROJ_DIR/models/vocab.txt"
```

---

## 8. Run the App

```bash
conda activate indicf5
export HSA_OVERRIDE_GFX_VERSION=11.0.0   # only needed for gfx1100 / RDNA3
python indicf5_app.py
```

Open **http://localhost:7860**

First run downloads the Vocos vocoder (~50 MB) automatically.

---

## Troubleshooting

### `ImportError: TorchCodec is required`
```bash
pip uninstall torchcodec -y
```
The app patches `torchaudio.load` to use soundfile instead.

### `RuntimeError: CUDA not available` / wrong device
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
python -c "import torch; print(torch.cuda.get_device_name())"
```

### `403 Forbidden` when downloading model
- Visit https://huggingface.co/ai4bharat/IndicF5 and accept the terms while logged in
- Re-run `huggingface-cli login` with a fresh token if still failing

### Audio is English, not Tamil
- Model loaded is `SWivid/F5-TTS` (English fallback) instead of `ai4bharat/IndicF5`
- Check startup logs for `[IndicF5] WARNING` — resolve the 403 above

### `TypeError: load_model() takes from 2 to 7 positional arguments but 8 given`
- Already patched in `indicf5_app.py` automatically — should not occur

---

## Directory Structure After Setup

```python
IndicF5-GUI/
├── indicf5_app.py
├── models/
│   ├── model.safetensors  -> ~/.cache/huggingface/hub/.../model.safetensors
│   └── vocab.txt          -> ~/.cache/huggingface/hub/.../checkpoints/vocab.txt
├── samples/
│   ├── hi_male.wav, ta_female.wav, ...
├── README.md
└── SETUP.md

~/.cache/huggingface/hub/
├── models--ai4bharat--IndicF5/      # 1.4 GB checkpoint
└── models--charactr--vocos-mel-24khz/  # ~50 MB vocoder
```
