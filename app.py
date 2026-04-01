
import os
import torch

# Patch PyTorch 2.6+ default behavior for trusted checkpoints
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from TTS.api import TTS
from inference.generate import load_model
from interface.gradio_app import launch_interface

os.environ["COQUI_TOS_AGREED"] = "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "best_model_1370.pth")
CONFIG_PATH = os.path.join(BASE_DIR, "checkpoints", "config.json")

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

# Load XTTS base model once so vocab.json is available locally
tts_base = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

MODEL_DIR = "/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/"
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.json")

if not os.path.exists(VOCAB_PATH):
    raise FileNotFoundError(f"Vocab not found: {VOCAB_PATH}")

model, config = load_model(
    checkpoint_path=CHECKPOINT_PATH,
    config_path=CONFIG_PATH,
    vocab_path=VOCAB_PATH,
)

launch_interface(model, config, share=True)
