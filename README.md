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
\`\`\`bash
pip install -r requirements.txt
apt-get install -y ffmpeg
\`\`\`

## Utilisation rapide

### 1. Préparer les données
\`\`\`python
from data.prepare_dataset import prepare_all
prepare_all(n_samples=650, output_dir="doda_darija")
\`\`\`

### 2. Finetuner (GPU T4 requis)
\`\`\`python
from training.finetune import finetune
finetune(
    data_path   = "doda_darija",
    model_dir   = "/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/",
    output_path = "outputs",
    epochs      = 5,
)
\`\`\`

### 3. Générer un audio
\`\`\`python
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
\`\`\`

### 4. Interface Gradio
\`\`\`python
from interface.gradio_app import launch_interface
launch_interface(model, config, share=True)
\`\`\`

## Dataset
DODa : https://huggingface.co/datasets/atlasia/DODa-audio-dataset

## Structure du projet
\`\`\`
xtts-darija/
├── config/           ← Configuration centralisée
├── data/             ← Préparation du dataset
├── training/         ← Script de finetuning
├── inference/        ← Génération audio
├── evaluation/       ← WER, CER, MOS
├── interface/        ← Interface Gradio
└── notebooks/        ← Demo Colab
\`\`\`



##  Exécuter le modèle sur une autre machine (sans re-fine-tuning)

###  Important
Le repository hébergé sur GitHub **ne contient pas les fichiers du modèle entraîné** (`best_model_1370.pth`, `config.json`) car ils sont trop volumineux

##  Pourquoi aucun fine-tuning n’est nécessaire

Le fine-tuning est effectué **une seule fois** pour générer les fichiers finaux :

- `best_model_1370.pth`
- `config.json`

Ensuite, le modèle peut être réutilisé à l’infini.

La fonction `load_model()` :
- charge la configuration XTTS  
- initialise le modèle  
- charge les poids entraînés (checkpoint)  
- charge le tokenizer / vocabulaire  
- envoie le modèle sur CPU ou GPU  

Donc lorsque vous exécutez `app.py`, le modèle est simplement **restauré**, pas réentraîné.

---

## Ce dont vous avez besoin
Pour exécuter le projet sur une autre machine :

1. Le repository du code  
2. Les dépendances installées  
3. Les fichiers du modèle entraîné :
   - `best_model_1370.pth`
   - `config.json`  
4. Le tokenizer XTTS (géré automatiquement)
   
## Étapes pour exécuter sur une autre machine / Colab

### 1) Installer les dépendances
```bash
!pip install -q coqui-tts==0.24.2 gradio soundfile numpy
!apt-get update -y && apt-get install -y ffmpeg

 ### 2)  monter Drive si tu gardes le modèle dans Drive
from google.colab import drive
drive.mount('/content/drive')
 3)  cloner le repo GitHub
%cd /content
!git clone https://github.com/ChaimaeHde/xtts-darija30minTest.git
%cd /content/xtts-darija30minTest
!ls
4) remettre les checkpoints
import os, shutil

MODEL_SOURCE_DIR = "/content/drive/MyDrive/xtts_darija_pfa"
CHECKPOINTS_DIR = "/content/xtts-darija30minTest/checkpoints"

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
shutil.copy2(os.path.join(MODEL_SOURCE_DIR, "best_model_1370.pth"), CHECKPOINTS_DIR)
shutil.copy2(os.path.join(MODEL_SOURCE_DIR, "config.json"), CHECKPOINTS_DIR)

print("checkpoints ready:", os.listdir(CHECKPOINTS_DIR))
5) lancer
!python app.py



## Auteurs
Projet de Fin d'Année — Synthèse Vocale Expressive pour le Darija Marocain
