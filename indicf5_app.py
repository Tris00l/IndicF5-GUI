# ruff: noqa: E402
import os
import re
import tempfile
from collections import OrderedDict
from importlib.resources import files as _pkg_files
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Force soundfile backend for torchaudio (torchcodec incompatible w/ ROCm) ──
import torchaudio as _torchaudio
try:
    _torchaudio.set_audio_backend("soundfile")
except Exception:
    # torchaudio 2.9+ removed set_audio_backend; patch load() directly
    import soundfile as _sf_patch

    _SF_UNSUPPORTED = {".webm", ".m4a", ".mp4", ".aac", ".wma", ".opus"}

    def _torchaudio_load_soundfile(path, *args, **kwargs):
        import subprocess, tempfile
        path = str(path)
        ext = os.path.splitext(path)[1].lower()
        if ext in _SF_UNSUPPORTED:
            # convert to wav via ffmpeg then load
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            subprocess.run(
                ["ffmpeg", "-y", "-i", path, "-ar", "24000", "-ac", "1", tmp_path],
                check=True, capture_output=True,
            )
            data, sr = _sf_patch.read(tmp_path, dtype="float32", always_2d=True)
            os.unlink(tmp_path)
        else:
            data, sr = _sf_patch.read(path, dtype="float32", always_2d=True)
        tensor = torch.from_numpy(data.T)  # (channels, samples)
        return tensor, sr

    _torchaudio.load = _torchaudio_load_soundfile

# ── Patch broken utils_infer.load_model before any f5_tts imports ─────────────
# The IndicF5 package ships with load_model that (a) has wrong signature and
# (b) has checkpoint loading commented out. Monkey-patch it to the correct form.
_APP_DIR       = Path(__file__).resolve().parent
_MODELS_DIR    = _APP_DIR / "models"
_INDICF5_REPO  = "ai4bharat/IndicF5"
_INDICF5_CKPT  = "model.safetensors"
_INDICF5_VOCAB = "checkpoints/vocab.txt"

def _get_indic_assets() -> tuple[str, str]:
    """Return (ckpt_path, vocab_path) for ai4bharat/IndicF5.
    Checks local models/ dir first, then HF hub, falls back to English."""
    local_ckpt  = _MODELS_DIR / "model.safetensors"
    local_vocab = _MODELS_DIR / "vocab.txt"
    if local_ckpt.exists() and local_vocab.exists():
        print(f"[IndicF5] model : {local_ckpt}")
        print(f"[IndicF5] vocab : {local_vocab}")
        return str(local_ckpt), str(local_vocab)

    try:
        ckpt   = hf_hub_download(repo_id=_INDICF5_REPO, filename=_INDICF5_CKPT)
        vocab  = hf_hub_download(repo_id=_INDICF5_REPO, filename=_INDICF5_VOCAB)
        print(f"[IndicF5] model : {ckpt}")
        print(f"[IndicF5] vocab : {vocab}")
        return ckpt, vocab
    except Exception as e:
        print(f"[IndicF5] WARNING: cannot load ai4bharat/IndicF5 — {e}")
        print("[IndicF5] Falling back to English model (Tamil will be wrong).")

    cache_dir = Path.home() / ".cache" / "indicf5"
    cache_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = cache_dir / "vocab_en.txt"
    if not vocab_path.exists():
        downloaded = hf_hub_download(
            repo_id="SWivid/F5-TTS", filename="F5TTS_Base/vocab.txt",
            cache_dir=str(cache_dir / "hf"),
        )
        import shutil
        shutil.copy(downloaded, str(vocab_path))
    return "", str(vocab_path)

def _get_vocab_path() -> str:
    _, vocab = _get_indic_assets()
    return vocab

def _patched_load_checkpoint(model, ckpt_path, device, dtype=None, use_ema=True):
    """load_checkpoint that also strips torch.compile()'s _orig_mod. prefix."""
    import torch as _torch
    from safetensors.torch import load_file as _load_sf
    if dtype is None:
        dtype = _torch.float32
    model = model.to(dtype)
    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        raw = _load_sf(ckpt_path, device=device)
    else:
        raw = _torch.load(ckpt_path, map_location=device, weights_only=True)

    if use_ema:
        if ckpt_type == "safetensors":
            raw = {"ema_model_state_dict": raw}
        state = {
            k.replace("ema_model.", "").replace("_orig_mod.", ""): v
            for k, v in raw["ema_model_state_dict"].items()
            if k not in ["initted", "step"] and not k.replace("ema_model.", "").replace("_orig_mod.", "").startswith("vocoder.")
        }
    else:
        if ckpt_type == "safetensors":
            raw = {"model_state_dict": raw}
        state = {
            k.replace("_orig_mod.", ""): v
            for k, v in raw["model_state_dict"].items()
            if not k.replace("_orig_mod.", "").startswith("vocoder.")
        }

    for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
        state.pop(key, None)

    model.load_state_dict(state)
    del raw, state
    _torch.cuda.empty_cache()
    return model.to(device)


def _patched_load_model(model_cls, model_cfg, ckpt_path, mel_spec_type="vocos",
                        vocab_file="", ode_method="euler", use_ema=True, device=None):
    import torch as _torch
    from f5_tts.model import CFM
    from f5_tts.model.utils import get_tokenizer
    from f5_tts.infer.utils_infer import (
        n_mel_channels, n_fft, hop_length, win_length, target_sample_rate,
    )
    if device is None:
        device = "cuda" if _torch.cuda.is_available() else "cpu"
    if not vocab_file:
        vocab_file = _get_vocab_path()
    vocab_char_map, vocab_size = get_tokenizer(vocab_file, "custom")
    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            n_mel_channels=n_mel_channels, target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(method=ode_method),
        vocab_char_map=vocab_char_map,
    ).to(device)
    dtype = _torch.float32 if mel_spec_type == "bigvgan" else None
    return _patched_load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

import f5_tts.infer.utils_infer as _utils_infer
_utils_infer.load_model = _patched_load_model
_utils_infer.load_checkpoint = _patched_load_checkpoint

# Now safe to import F5TTS (it will use patched load_model)
from f5_tts.api import F5TTS

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Sample audio directory ──────────────────────────────────────────────────────
_SAMPLES_DIR = _APP_DIR / "samples"
DEFAULT_REF  = str(_SAMPLES_DIR / "ta_female.wav")

# ── Model singletons ────────────────────────────────────────────────────────────
_tts_instances: dict[str, F5TTS] = {}
chat_model_state     = None
chat_tokenizer_state = None

_INDIC_CKPT, _INDIC_VOCAB = None, None

def _load_indic_assets_once():
    global _INDIC_CKPT, _INDIC_VOCAB
    if _INDIC_CKPT is None:
        _INDIC_CKPT, _INDIC_VOCAB = _get_indic_assets()

def _get_tts(model_type="F5-TTS", ckpt_file="", vocab_file="") -> F5TTS:
    _load_indic_assets_once()
    # Use IndicF5 assets unless caller explicitly overrides
    effective_ckpt  = ckpt_file  or _INDIC_CKPT  or ""
    effective_vocab = vocab_file or _INDIC_VOCAB  or ""
    key = f"{model_type}:{effective_ckpt}"
    if key not in _tts_instances:
        gr.Info(f"Loading {model_type} model… (first run downloads ~1-2 GB)")
        _tts_instances[key] = F5TTS(
            model_type=model_type,
            ckpt_file=effective_ckpt,
            vocab_file=effective_vocab,
            device=DEVICE,
        )
    return _tts_instances[key]

# ── Core inference ─────────────────────────────────────────────────────────────
def infer(ref_audio_orig, ref_text, gen_text, model_choice,
          remove_silence, cross_fade_duration=0.15, speed=1.0,
          ckpt_file="", vocab_file="", show_info=gr.Info):

    if isinstance(model_choice, list) and model_choice[0] == "Custom":
        model_type = "F5-TTS"
        ckpt_file  = model_choice[1]
        vocab_file = model_choice[2]
    else:
        model_type = model_choice  # "F5-TTS" or "E2-TTS"

    tts = _get_tts(model_type, ckpt_file, vocab_file)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as spec_tmp:
        spec_path = spec_tmp.name

    wav, sr, _spect = tts.infer(
        ref_file=ref_audio_orig,
        ref_text=ref_text,
        gen_text=gen_text,
        remove_silence=remove_silence,
        speed=speed,
        cross_fade_duration=cross_fade_duration,
        file_spect=spec_path,
        show_info=show_info,
    )

    return (sr, wav), spec_path, ref_text

# ── Voice-chat helpers ─────────────────────────────────────────────────────────
def generate_response(messages, model, tokenizer):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    ids = model.generate(**inputs, max_new_tokens=512, temperature=0.7, top_p=0.95)
    ids = [o[len(i):] for i, o in zip(inputs.input_ids, ids)]
    return tokenizer.batch_decode(ids, skip_special_tokens=True)[0]

# ── CSS ────────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-base:     #07080f;
    --bg-surface:  #0d0f1c;
    --bg-card:     #131627;
    --bg-card2:    #181c30;
    --border:      #252840;
    --border-hi:   #363a5c;
    --amber:       #f59e0b;
    --amber-dim:   #d97706;
    --amber-soft:  rgba(245,158,11,0.12);
    --amber-glow:  rgba(245,158,11,0.25);
    --cyan:        #22d3ee;
    --cyan-dim:    #0891b2;
    --cyan-soft:   rgba(34,211,238,0.10);
    --cyan-glow:   rgba(34,211,238,0.22);
    --violet:      #a78bfa;
    --violet-soft: rgba(167,139,250,0.12);
    --green:       #34d399;
    --text:        #e8eaf6;
    --text-muted:  #7e84a3;
    --text-dim:    #4a4f6a;
    --radius:      10px;
    --radius-lg:   16px;
}

/* ── Base ─────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container, .main {
    background: var(--bg-base) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* subtle dot-grid background */
body::after {
    content: "";
    position: fixed;
    inset: 0;
    background-image: radial-gradient(circle, rgba(245,158,11,0.045) 1px, transparent 1px);
    background-size: 28px 28px;
    pointer-events: none;
    z-index: 0;
}

/* ── Header ───────────────────────────────────────────────── */
.app-header {
    text-align: center;
    padding: 48px 24px 32px;
    position: relative;
    background: linear-gradient(180deg, #0c0c22 0%, #0a0b14 60%, var(--bg-base) 100%);
    border-bottom: 1px solid var(--border);
    overflow: hidden;
}

.app-header::before {
    content: "";
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 70% 60% at 50% -5%,  rgba(245,158,11,0.18) 0%, transparent 65%),
        radial-gradient(ellipse 40% 40% at 20% 50%,  rgba(34,211,238,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 40% 40% at 80% 50%,  rgba(167,139,250,0.07) 0%, transparent 60%);
    pointer-events: none;
}

.app-header-line {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--amber), var(--cyan), var(--violet), transparent);
    opacity: 0.6;
}

.app-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 3rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
    color: #f59e0b !important;
    -webkit-text-fill-color: #f59e0b !important;
    background: none !important;
    text-shadow: 0 0 30px rgba(245,158,11,0.6), 0 0 60px rgba(245,158,11,0.25) !important;
    margin: 0 0 6px !important;
    line-height: 1.05 !important;
    position: relative !important;
    display: block !important;
}

.ascii-title {
    font-family: 'JetBrains Mono', monospace, 'Courier New', Courier !important;
    font-size: clamp(0.38rem, 1.1vw, 0.72rem) !important;
    line-height: 1.25 !important;
    letter-spacing: 0.02em !important;
    margin: 0 auto 8px !important;
    padding: 0 !important;
    background: none !important;
    border: none !important;
    display: block !important;
    text-align: center !important;
    white-space: pre !important;
    overflow-x: auto !important;
    -webkit-text-fill-color: unset !important;
}

.app-subtitle {
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-muted);
    font-size: 0.78rem;
    margin-top: 6px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

@keyframes shimmer {
    0%   { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}

/* ── Tabs ─────────────────────────────────────────────────── */
.tab-nav {
    background: var(--bg-surface) !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 0 20px !important;
    gap: 2px !important;
}

.tab-nav button {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-muted) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 14px 22px !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
}

.tab-nav button:hover {
    color: var(--amber) !important;
    border-bottom-color: rgba(245,158,11,0.35) !important;
    background: rgba(245,158,11,0.04) !important;
}

.tab-nav button.selected {
    color: var(--amber) !important;
    border-bottom-color: var(--amber) !important;
    background: rgba(245,158,11,0.06) !important;
    font-weight: 600 !important;
}

/* ── All blocks get card treatment ───────────────────────── */
.block, .form {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    transition: border-color 0.2s ease !important;
}

.block:hover {
    border-color: var(--border-hi) !important;
}

/* ── Inputs ───────────────────────────────────────────────── */
input, textarea, select,
.block textarea,
.block input[type=text],
.block input[type=number] {
    font-family: 'DM Sans', sans-serif !important;
    background: var(--bg-card2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

input:focus, textarea:focus {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 3px var(--amber-soft), 0 0 16px var(--amber-glow) !important;
    outline: none !important;
}


label, .label-wrap span {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--text-muted) !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

/* ── Dropdown ─────────────────────────────────────────────── */
.multiselect, .wrap-inner, .dropdown-arrow {
    background: var(--bg-card2) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}

.multiselect:hover, .wrap-inner:hover {
    border-color: var(--amber-dim) !important;
}

/* ── Radio buttons (Female / Male pills) ─────────────────── */
.radio-group {
    gap: 8px !important;
}

.radio-group label {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 8px 14px !important;
    cursor: pointer !important;
    transition: all 0.18s ease !important;
    text-transform: none !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.01em !important;
    color: var(--text-muted) !important;
    font-family: 'DM Sans', sans-serif !important;
}

.radio-group label:hover {
    border-color: var(--amber-dim) !important;
    background: var(--amber-soft) !important;
    color: var(--amber) !important;
}

.radio-group input[type=radio]:checked + span,
.radio-group label:has(input:checked) {
    border-color: var(--amber) !important;
    background: var(--amber-soft) !important;
    color: var(--amber) !important;
    box-shadow: 0 0 10px var(--amber-glow) !important;
}

/* ── Sliders ──────────────────────────────────────────────── */
input[type=range] {
    accent-color: var(--amber) !important;
}

/* ── Buttons ──────────────────────────────────────────────── */
button.primary, .gr-button-primary {
    font-family: 'DM Sans', sans-serif !important;
    background: linear-gradient(135deg, var(--amber-dim) 0%, #b45309 50%, var(--amber) 100%) !important;
    background-size: 200% !important;
    color: #0a0a0a !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    font-size: 0.9rem !important;
    padding: 13px 32px !important;
    box-shadow: 0 0 20px var(--amber-glow), 0 4px 20px rgba(0,0,0,0.4) !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
    text-transform: uppercase !important;
}

button.primary:hover, .gr-button-primary:hover {
    background-position: right center !important;
    box-shadow: 0 0 36px var(--amber-glow), 0 0 60px rgba(245,158,11,0.2), 0 4px 24px rgba(0,0,0,0.5) !important;
    transform: translateY(-2px) !important;
}

button.secondary, .gr-button-secondary {
    font-family: 'DM Sans', sans-serif !important;
    background: transparent !important;
    color: var(--cyan) !important;
    border: 1px solid rgba(34,211,238,0.35) !important;
    border-radius: var(--radius) !important;
    font-weight: 500 !important;
    padding: 8px 18px !important;
    transition: all 0.2s ease !important;
}

button.secondary:hover, .gr-button-secondary:hover {
    background: var(--cyan-soft) !important;
    border-color: var(--cyan) !important;
    box-shadow: 0 0 12px var(--cyan-glow) !important;
}

/* ── Audio ────────────────────────────────────────────────── */
.audio-player, gradio-audio, .gr-audio,
.waveform-container, audio {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* ── Accordion ────────────────────────────────────────────── */
.gr-accordion, .accordion {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

.accordion > .label-wrap {
    padding: 12px 16px !important;
    border-bottom: 1px solid var(--border) !important;
}

/* ── Checkbox ─────────────────────────────────────────────── */
input[type=checkbox] { accent-color: var(--amber) !important; }

/* ── Image ────────────────────────────────────────────────── */
img {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
}

/* ── Progress / generating ────────────────────────────────── */
.generating, .eta-bar {
    background: linear-gradient(90deg, var(--bg-card), var(--amber-soft)) !important;
    border-color: var(--amber-dim) !important;
}

/* ── Chatbot ──────────────────────────────────────────────── */
.chatbot, .message-wrap {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

.message.user { background: var(--amber-soft) !important; border-radius: 8px !important; }
.message.bot  { background: var(--violet-soft) !important; border-radius: 8px !important; }

/* ── Divider ──────────────────────────────────────────────── */
hr { border-color: var(--border) !important; }

/* ── Model badge ──────────────────────────────────────────── */
.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--amber-soft);
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 6px;
    padding: 4px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--amber);
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 3px;
}

.model-badge.violet {
    background: var(--violet-soft);
    border-color: rgba(167,139,250,0.3);
    color: var(--violet);
}

/* ── Pulse dot ────────────────────────────────────────────── */
.pulse-dot::before {
    content: "●";
    color: var(--green);
    font-size: 0.45rem;
    animation: pulse 2s ease-in-out infinite;
    margin-right: 6px;
    vertical-align: middle;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.2; }
}

/* ── Section labels ───────────────────────────────────────── */
.section-label {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--amber);
    margin-bottom: 10px;
    opacity: 0.85;
}

.section-label::before, .section-label::after {
    content: "";
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(245,158,11,0.3));
}

.section-label::before {
    background: linear-gradient(270deg, transparent, rgba(245,158,11,0.3));
    flex: 0 0 20px;
}

/* ── Language chips ───────────────────────────────────────── */
.lang-chip {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    color: var(--text-dim);
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 10px;
    letter-spacing: 0.04em;
    transition: all 0.18s ease;
}

.lang-chip:hover {
    color: var(--cyan);
    border-color: rgba(34,211,238,0.3);
    background: var(--cyan-soft);
}

/* ── Scrollbar ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--amber-dim); }
"""

# ── Multi-style helpers ────────────────────────────────────────────────────────
def parse_speechtypes_text(gen_text):
    pattern = r"\{(.*?)\}"
    tokens = re.split(pattern, gen_text)
    segments = []
    current_style = "Regular"
    for i, token in enumerate(tokens):
        if i % 2 == 0:
            if token.strip():
                segments.append({"style": current_style, "text": token.strip()})
        else:
            current_style = token.strip()
    return segments

# Language → (code, has_female, has_male)
_LANGUAGES = {
    "Tamil":     ("ta",  True,  True),
    "Hindi":     ("hi",  True,  True),
    "Telugu":    ("te",  True,  True),
    "Kannada":   ("kn",  True,  True),
    "Malayalam": ("ml",  True,  True),
    "Marathi":   ("mr",  True,  True),
    "Gujarati":  ("gu",  True,  True),
    "Bengali":   ("bn",  False, True),
    "Assamese":  ("as",  False, False),
    "Odia":      ("or",  False, False),
    "Punjabi":   ("pa",  False, False),
}
_LANG_NAMES = list(_LANGUAGES.keys())

def _sample_path(lang: str, gender: str) -> str | None:
    code, has_f, has_m = _LANGUAGES[lang]
    if gender == "Female" and not has_f:
        if has_m:
            gender = "Male"
        else:
            return None
    if gender == "Male" and not has_m:
        if has_f:
            gender = "Female"
        else:
            return None
    p = _SAMPLES_DIR / f"{code}_{gender.lower()}.wav"
    return str(p) if p.exists() else None

_PRESET_TRANSCRIPTS: dict[str, str] = {}

# ── App ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="IndicF5 TTS") as app:

    # Header
    gr.HTML("""
    <div class="app-header">
        <div class="app-header-line"></div>
        <pre class="ascii-title"><span style="color:#FF9933">██╗███╗   ██╗██████╗ ██╗ ██████╗███████╗███████╗
██║████╗  ██║██╔══██╗██║██╔════╝██╔════╝██╔════╝</span>
<span style="color:#f0f0f0">██║██╔██╗ ██║██║  ██║██║██║     █████╗  ███████╗
██║██║╚██╗██║██║  ██║██║██║     ██╔══╝  ╚════██║</span>
<span style="color:#138808">██║██║ ╚████║██████╔╝██║╚██████╗██║     ███████║
╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝ ╚═════╝╚═╝     ╚══════╝</span></pre>
        <p class="app-subtitle">AI4Bharat &nbsp;·&nbsp; Flow-Matching TTS &nbsp;·&nbsp; 11 Indic Languages</p>
        <div style="margin-top:16px; display:flex; justify-content:center; flex-wrap:wrap; gap:6px;">
            <span class="model-badge pulse-dot">Synthesize</span>
            <span class="model-badge violet">Multi-Style</span>
            <span class="model-badge">Voice Chat</span>
            <span class="model-badge violet">IndicF5 Model</span>
        </div>
        <div style="margin-top:12px; display:flex; justify-content:center; flex-wrap:wrap; gap:5px;">
            <span class="lang-chip">Tamil</span>
            <span class="lang-chip">Hindi</span>
            <span class="lang-chip">Telugu</span>
            <span class="lang-chip">Kannada</span>
            <span class="lang-chip">Malayalam</span>
            <span class="lang-chip">Marathi</span>
            <span class="lang-chip">Gujarati</span>
            <span class="lang-chip">Bengali</span>
            <span class="lang-chip">Assamese</span>
            <span class="lang-chip">Odia</span>
            <span class="lang-chip">Punjabi</span>
        </div>
    </div>
    """)

    tts_model_choice = gr.State("F5-TTS")

    with gr.Tabs():

        # ── Tab 1: Basic TTS ───────────────────────────────────────────────────
        with gr.Tab("⚡  Synthesize"):
            with gr.Row(equal_height=False):

                # Left column — inputs
                with gr.Column(scale=5):
                    gr.HTML('<div class="section-label">Reference Voice</div>')
                    with gr.Row():
                        lang_dropdown = gr.Dropdown(
                            choices=_LANG_NAMES,
                            value="Tamil",
                            label="Language",
                            scale=3,
                        )
                        gender_radio = gr.Radio(
                            choices=["Female", "Male"],
                            value="Female",
                            label="Gender",
                            scale=2,
                        )
                    ref_audio_preview = gr.Audio(
                        label="Preset Preview",
                        type="filepath",
                        value=DEFAULT_REF,
                        interactive=False,
                    )
                    ref_audio_upload = gr.Audio(
                        label="Custom Voice Override  (Record or Upload — optional)",
                        type="filepath",
                        sources=["upload", "microphone"],
                        interactive=True,
                    )
                    ref_text_input = gr.State(value=_PRESET_TRANSCRIPTS.get("ta_female", ""))
                    gr.HTML('<div class="section-label" style="margin-top:16px;">Text to Synthesize</div>')
                    gen_text_input = gr.Textbox(
                        label="",
                        lines=5,
                        placeholder="Enter text to convert to speech…",
                    )
                    generate_btn = gr.Button("▶  Generate", variant="primary", size="lg")

                    with gr.Accordion("⚙  Advanced", open=False):
                        remove_silence = gr.Checkbox(label="Remove Silences", value=True)
                        speed_slider = gr.Slider(
                            0.3, 2.0, value=1.0, step=0.05,
                            label="Speed",
                            info="1.0 = normal"
                        )
                        cross_fade_slider = gr.Slider(
                            0.0, 1.0, value=0.15, step=0.01,
                            label="Cross-Fade Duration (s)"
                        )

                # Right column — outputs
                with gr.Column(scale=5):
                    gr.HTML('<div class="section-label">Output</div>')
                    audio_output = gr.Audio(label="Synthesized Audio", interactive=False)
                    spectrogram_output = gr.Image(
                        label="Mel Spectrogram",
                        show_label=True,
                        interactive=False,
                    )

            def basic_tts(ref_audio_custom, ref_audio_preset, ref_text, gen_text, model, remove_sil, speed, cross_fade):
                ref_audio = ref_audio_custom if ref_audio_custom else ref_audio_preset
                if not ref_audio:
                    raise gr.Error("Reference audio required. Upload or record a voice sample (5–15 sec).")
                if not gen_text or not gen_text.strip():
                    raise gr.Error("Enter text to synthesize.")
                audio_out, spec_path, ref_text_out = infer(
                    ref_audio, ref_text, gen_text, model,
                    remove_sil, cross_fade, speed,
                )
                return audio_out, spec_path, ref_text_out

            def on_voice_change(lang, gender):
                code = _LANGUAGES[lang][0]
                path = _sample_path(lang, gender)
                transcript = _PRESET_TRANSCRIPTS.get(f"{code}_{gender.lower()}", "")
                return path, transcript  # None path clears the audio player

            lang_dropdown.change(on_voice_change, inputs=[lang_dropdown, gender_radio],
                                 outputs=[ref_audio_preview, ref_text_input])
            gender_radio.change(on_voice_change, inputs=[lang_dropdown, gender_radio],
                                outputs=[ref_audio_preview, ref_text_input])

            generate_btn.click(
                basic_tts,
                inputs=[ref_audio_upload, ref_audio_preview, ref_text_input, gen_text_input,
                        tts_model_choice, remove_silence, speed_slider, cross_fade_slider],
                outputs=[audio_output, spectrogram_output, ref_text_input],
            )

        # ── Tab 2: Multi-Style TTS ─────────────────────────────────────────────
        with gr.Tab("🎭  Multi-Style"):
            gr.HTML("""
            <div style="padding:12px 0 4px; color:var(--text-muted); font-size:0.85rem;">
                Tag different speech styles inline:
                <code style="background:var(--bg-card);padding:2px 6px;border-radius:4px;color:var(--cyan);">
                  {Style} text here {Emotion} more text
                </code>
            </div>""")

            max_speech_types = 5

            gen_text_input_ms = gr.Textbox(
                label="Script with Style Tags",
                lines=6,
                placeholder="{Regular} Hello, how can I help you today?\n{Happy} That's wonderful news!\n{Sad} I'm sorry to hear that.",
            )

            speech_type_rows, speech_type_names, speech_type_audios, speech_type_ref_texts = [], [], [], []
            speech_type_delete_btns, speech_type_insert_btns = [], []
            speech_type_count = gr.State(value=1)

            with gr.Column():
                for i in range(max_speech_types):
                    with gr.Row(visible=(i == 0)) as row:
                        with gr.Column(scale=2):
                            name = gr.Textbox(
                                value="Regular" if i == 0 else "",
                                label=f"Style {i+1} Name",
                                placeholder="e.g. Happy, Sad, Whispering…",
                            )
                        with gr.Column(scale=3):
                            audio = gr.Audio(
                                label=f"Style {i+1} Reference",
                                type="filepath",
                                sources=["upload", "microphone"],
                            )
                        with gr.Column(scale=3):
                            ref_text = gr.Textbox(label=f"Style {i+1} Ref Text", lines=2)
                        with gr.Column(scale=1, min_width=80):
                            ins_btn = gr.Button(f"Insert ↑", variant="secondary", size="sm")
                            del_btn = gr.Button("✕", variant="secondary", size="sm")
                    speech_type_rows.append(row)
                    speech_type_names.append(name)
                    speech_type_audios.append(audio)
                    speech_type_ref_texts.append(ref_text)
                    speech_type_insert_btns.append(ins_btn)
                    speech_type_delete_btns.append(del_btn)

            with gr.Row():
                add_style_btn = gr.Button("＋ Add Style", variant="secondary")

            with gr.Accordion("⚙  Advanced", open=False):
                remove_silence_ms = gr.Checkbox(label="Remove Silences", value=True)

            generate_ms_btn = gr.Button("▶  Generate Multi-Style", variant="primary", size="lg")
            audio_output_ms = gr.Audio(label="Output", interactive=False)

            def add_speech_type(count):
                count = min(count + 1, max_speech_types)
                return [count] + [gr.update(visible=(i < count)) for i in range(max_speech_types)]

            add_style_btn.click(
                add_speech_type, inputs=speech_type_count,
                outputs=[speech_type_count] + speech_type_rows,
            )

            def make_delete_fn(idx):
                def _delete(count):
                    count = max(1, count - 1)
                    return [count] + [gr.update(visible=(i < count)) for i in range(max_speech_types)]
                return _delete

            for i, del_btn in enumerate(speech_type_delete_btns):
                del_btn.click(make_delete_fn(i), inputs=speech_type_count,
                              outputs=[speech_type_count] + speech_type_rows)

            def make_insert_fn(idx):
                def _insert(gen_text, name):
                    return gen_text + f"{{{name}}} "
                return _insert

            for i, ins_btn in enumerate(speech_type_insert_btns):
                ins_btn.click(make_insert_fn(i),
                              inputs=[gen_text_input_ms, speech_type_names[i]],
                              outputs=gen_text_input_ms)

            def generate_multistyle(gen_text, *args):
                names    = args[:max_speech_types]
                audios   = args[max_speech_types:2*max_speech_types]
                ref_txts = args[2*max_speech_types:3*max_speech_types]
                remove_sil = args[3*max_speech_types]

                speech_types = OrderedDict()
                for name, audio, rtxt in zip(names, audios, ref_txts):
                    if name and audio:
                        speech_types[name] = {"audio": audio, "ref_text": rtxt}

                if not speech_types:
                    raise gr.Error("Add at least one style with a reference audio.")

                segments = parse_speechtypes_text(gen_text)
                if not segments:
                    raise gr.Error("No text segments found.")

                generated = []
                current_style = "Regular"
                for seg in segments:
                    style = seg["style"]
                    if style in speech_types:
                        current_style = style
                    ref_audio = speech_types[current_style]["audio"]
                    ref_text  = speech_types[current_style].get("ref_text", "")
                    audio_out, _, ref_text_out = infer(
                        ref_audio, ref_text, seg["text"], "F5-TTS", remove_sil, show_info=print
                    )
                    sr, wave = audio_out
                    generated.append(wave)
                    speech_types[current_style]["ref_text"] = ref_text_out

                final = np.concatenate(generated)
                return (sr, final)

            generate_ms_btn.click(
                generate_multistyle,
                inputs=[gen_text_input_ms]
                    + speech_type_names
                    + speech_type_audios
                    + speech_type_ref_texts
                    + [remove_silence_ms],
                outputs=audio_output_ms,
            )

        # ── Tab 3: Voice Chat ──────────────────────────────────────────────────
        with gr.Tab("💬  Voice Chat"):
            gr.HTML("""
            <div style="padding:12px 0 4px; color:var(--text-muted); font-size:0.85rem;">
                Converse with an AI that responds in your reference voice.
                Upload a voice sample, load the chat model, then speak or type.
            </div>""")

            load_chat_btn = gr.Button("⬡  Load Chat Model (Qwen2.5-3B)", variant="secondary")
            chat_container = gr.Column(visible=False)

            def load_chat_model():
                global chat_model_state, chat_tokenizer_state
                if chat_model_state is None:
                    gr.Info("Loading Qwen2.5-3B…")
                    model_name = "Qwen/Qwen2.5-3B-Instruct"
                    chat_model_state     = AutoModelForCausalLM.from_pretrained(
                        model_name, torch_dtype="auto", device_map="auto")
                    chat_tokenizer_state = AutoTokenizer.from_pretrained(model_name)
                    gr.Info("Chat model ready.")
                return gr.update(visible=False), gr.update(visible=True)

            load_chat_btn.click(load_chat_model, outputs=[load_chat_btn, chat_container])

            with chat_container:
                with gr.Row():
                    with gr.Column(scale=4):
                        ref_audio_chat    = gr.Audio(label="Reference Voice", type="filepath",
                                                     sources=["upload", "microphone"])
                        system_prompt_chat = gr.Textbox(
                            label="AI Persona",
                            value="You are not an AI assistant, you are whoever the user says you are. Stay in character. Keep responses concise since they will be spoken out loud.",
                            lines=3,
                        )
                        with gr.Accordion("⚙  Advanced", open=False):
                            ref_text_chat     = gr.State(value="")
                            remove_silence_vc = gr.Checkbox(label="Remove Silences", value=True)

                    with gr.Column(scale=6):
                        chatbot_interface  = gr.Chatbot(label="Conversation", height=380)
                        audio_output_chat  = gr.Audio(label="AI Voice Response", autoplay=True,
                                                      interactive=False)
                        with gr.Row():
                            audio_input_chat = gr.Audio(
                                sources=["microphone"], type="filepath", label="Speak", scale=4
                            )
                            text_input_chat  = gr.Textbox(label="Or Type", scale=5, lines=1)
                            send_btn_chat    = gr.Button("Send", variant="primary", scale=1)
                        clear_btn_chat = gr.Button("Clear Conversation", variant="secondary")

                conversation_state = gr.State(
                    value=[{"role": "system",
                            "content": "You are not an AI assistant, you are whoever the user says you are. Stay in character. Keep responses concise since they will be spoken out loud."}]
                )

                def process_audio_input(audio_path, text, history, conv_state):
                    if not audio_path and not (text and text.strip()):
                        return history, conv_state
                    if audio_path:
                        from f5_tts.infer.utils_infer import transcribe
                        user_text = transcribe(audio_path)
                    else:
                        user_text = text.strip()
                    if not user_text:
                        return history, conv_state
                    conv_state = conv_state + [{"role": "user", "content": user_text}]
                    history    = history + [[user_text, None]]
                    return history, conv_state

                def generate_audio_response(history, ref_audio, ref_text, remove_sil):
                    if not history or history[-1][1] is not None:
                        return None, ref_text
                    global chat_model_state, chat_tokenizer_state
                    if chat_model_state is None:
                        return None, ref_text
                    response = generate_response(
                        [{"role": "system", "content": "You are a helpful assistant. Keep responses concise."}]
                        + [{"role": "user" if i % 2 == 0 else "assistant", "content": m}
                           for pair in history for i, m in enumerate(pair) if m],
                        chat_model_state, chat_tokenizer_state,
                    )
                    history[-1][1] = response
                    if not ref_audio:
                        return None, ref_text
                    audio_out, _, ref_text_out = infer(ref_audio, ref_text, response, "F5-TTS", remove_sil)
                    return audio_out, ref_text_out

                def clear_conversation():
                    return [], [{"role": "system",
                                 "content": "You are not an AI assistant, you are whoever the user says you are. Stay in character. Keep responses concise."}]

                def update_system_prompt(new_prompt):
                    return [], [{"role": "system", "content": new_prompt}]

                system_prompt_chat.change(update_system_prompt,
                    inputs=system_prompt_chat,
                    outputs=[chatbot_interface, conversation_state])

                for trigger, inputs_extra in [
                    (audio_input_chat.stop_recording, [audio_input_chat, text_input_chat]),
                    (text_input_chat.submit,           [audio_input_chat, text_input_chat]),
                    (send_btn_chat.click,               [audio_input_chat, text_input_chat]),
                ]:
                    trigger(
                        process_audio_input,
                        inputs=inputs_extra + [chatbot_interface, conversation_state],
                        outputs=[chatbot_interface, conversation_state],
                    ).then(
                        generate_audio_response,
                        inputs=[chatbot_interface, ref_audio_chat, ref_text_chat, remove_silence_vc],
                        outputs=[audio_output_chat, ref_text_chat],
                    ).then(lambda: None, None, audio_input_chat)

                clear_btn_chat.click(clear_conversation,
                    outputs=[chatbot_interface, conversation_state])

        # ── Tab 4: About ───────────────────────────────────────────────────────
        with gr.Tab("ℹ  About"):
            gr.HTML("""
            <div style="max-width:680px; margin:32px auto; color:var(--text-muted); line-height:1.8;">
                <h2 style="color:var(--text); font-size:1.4rem; font-weight:700; margin-bottom:16px;">
                    IndicF5 — Flow-Matching TTS for Indian Languages
                </h2>
                <p>
                    Built on <strong style="color:var(--cyan);">F5-TTS</strong> (flow-matching diffusion transformer),
                    fine-tuned by <strong style="color:var(--violet);">AI4Bharat</strong> for Indic scripts and phonology.
                </p>
                <hr style="border-color:var(--border); margin:20px 0;"/>
                <h3 style="color:var(--text); font-size:1rem; font-weight:600; margin-bottom:8px;">Usage Tips</h3>
                <ul style="padding-left:20px;">
                    <li>Reference audio: 5–15 seconds, clean, single speaker.</li>
                    <li>Leave <em>Reference Transcript</em> blank — Whisper auto-transcribes.</li>
                    <li>Lower <strong>Speed</strong> for slower, clearer output.</li>
                    <li>Multi-Style: wrap segments in <code style="color:var(--cyan);">{StyleName}</code> tags.</li>
                </ul>
                <hr style="border-color:var(--border); margin:20px 0;"/>
                <h3 style="color:var(--text); font-size:1rem; font-weight:600; margin-bottom:8px;">Environment</h3>
                <p class="pulse-dot" style="display:inline-flex; align-items:center;">
                    ROCm / AMD GPU (indicf5 conda env)
                </p>
                <hr style="border-color:var(--border); margin:20px 0;"/>
                <p style="font-size:0.8rem;">
                    <a href="https://github.com/ai4bharat/IndicF5"
                       style="color:var(--cyan); text-decoration:none;">
                        github.com/ai4bharat/IndicF5
                    </a>
                    &nbsp;·&nbsp;
                    <a href="https://github.com/SWivid/F5-TTS"
                       style="color:var(--violet); text-decoration:none;">
                        F5-TTS
                    </a>
                </p>
            </div>
            """)

if __name__ == "__main__":
    app.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Base(),
        css=CSS,
        allowed_paths=[str(_SAMPLES_DIR)],
    )
