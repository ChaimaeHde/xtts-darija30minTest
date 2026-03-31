"""
Définition explicite du modèle XTTS-v2 à entraîner.

Pourquoi GPTTrainer et pas Trainer générique ?
→ XTTS-v2 lève NotImplementedError sur train_step/eval_step/get_criterion
→ Il faut obligatoirement GPTTrainer de TTS.tts.layers.xtts.trainer.gpt_trainer

Pourquoi BaseDatasetConfig et pas un dict ?
→ load_tts_samples accède aux attributs en dot notation (dataset.meta_file_attn_mask)
→ Un dict lève AttributeError
"""

import os

# ── Chemins base model ──────────────────────────────────────────
BASE_MODEL_DIR = "/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2"

MODEL_FILES = {
    "config"    : os.path.join(BASE_MODEL_DIR, "config.json"),
    "vocab"     : os.path.join(BASE_MODEL_DIR, "vocab.json"),
    "dvae"      : os.path.join(BASE_MODEL_DIR, "dvae.pth"),
    "mel_stats" : os.path.join(BASE_MODEL_DIR, "mel_stats.pth"),
    "model"     : os.path.join(BASE_MODEL_DIR, "model.pth"),
}

# ── Paramètres du modèle ────────────────────────────────────────
MODEL_CONFIG = {
    "model_name"         : "tts_models/multilingual/multi-dataset/xtts_v2",
    "language"           : "ar",
    "use_cuda"           : True,
    "mixed_precision"    : True,   # fp16 — obligatoire sur T4 pour tenir en mémoire
}

# ── Paramètres de finetuning ────────────────────────────────────
TRAINING_CONFIG = {
    "epochs"             : 5,
    "batch_size"         : 2,
    "grad_accum_steps"   : 126,    # batch effectif = 252 (recommandé par Coqui)
    "learning_rate"      : 5e-6,
    "output_path"        : "/content/pfa_outputs",
    "run_name"           : "xtts_darija_pfa",
}

# ── Dataset ─────────────────────────────────────────────────────
DATASET_CONFIG = {
    "dataset_name"       : "doda_f1",
    "path"               : "/content/doda_f1_30min/",
    "meta_file_train"    : "train.csv",
    "formatter"          : "ljspeech",
    "language"           : "ar",
}

# ── Sorties ─────────────────────────────────────────────────────
OUTPUT_CONFIG = {
    "output_dir"         : "/content/pfa_outputs",
    "best_model"         : "/content/pfa_outputs/best_model_1370.pth",
    "drive_backup"       : "/content/drive/MyDrive/xtts_darija_pfa/",
    "hf_model_repo"      : "chaimaehde/xtts-darija",
    "hf_space_repo"      : "chaimaehde/tts-darija-marocain",
}
