# XTTS-v2 Darija Marocain — Fine-tuning Pipeline

Système de synthèse vocale expressive pour le dialecte marocain (Darija)
basé sur XTTS-v2 avec fine-tuning sur le dataset DODa.

## Résultats (pilote 30 min)

| Métrique | Valeur | Interprétation |
|---|---|---|
| WER moyen | 82.04% | Limité par Whisper/Darija |
| CER moyen | 38.49% | Phonétique correcte |
| MOS global | 2.85 / 5 | Acceptable pour 30 min |
| Loss mel_ce finale | 4.246 | Convergence partielle |

## Installation
```bash
pip install -r requirements.txt
apt-get install -y ffmpeg
```

## Utilisation rapide

### 1. Préparer les données
```python
from data.prepare_dataset import prepare_all
prepare_all(n_samples=650, output_dir="doda_darija")
```

### 2. Finetuner (GPU T4 requis)
```python
from training.finetune import finetune
finetune(
    data_path   = "doda_darija",
    model_dir   = "/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/",
    output_path = "outputs",
    epochs      = 5,
)
```

### 3. Générer un audio
```python
from inference.generate import load_model, generate
model, config = load_model(
    checkpoint_path = "outputs/best_model.pth",
    config_path     = "outputs/config.json",
    vocab_path      = "vocab.json",
)
generate(model, config,
    text        = "مرحبا، كيف داير؟",
    speaker_wav = "ref.wav",
    output_path = "output.wav",
)
```

### 4. Interface Gradio
```python
from interface.gradio_app import launch_interface
launch_interface(model, config, share=True)
```

## Demo
- HuggingFace Spaces : https://huggingface.co/spaces/chaimaehde/tts-darija-marocain
- Modèle finetuné   : https://huggingface.co/chaimaehde/xtts-darija

## Dataset
DODa : https://huggingface.co/datasets/atlasia/DODa-audio-dataset

## Structure du projet
```
xtts-darija/
├── config/           ← Configuration centralisée
├── data/             ← Préparation du dataset
├── training/         ← Script de finetuning
├── inference/        ← Génération audio
├── evaluation/       ← WER, CER, MOS
├── interface/        ← Interface Gradio
└── notebooks/        ← Demo Colab
```

## Auteurs
Projet de Fin d\'Année — Synthèse Vocale Expressive pour le Darija Marocain
