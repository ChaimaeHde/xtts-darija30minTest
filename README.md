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
```bash
from data.prepare_dataset import prepare_all
prepare_all(n_samples=650, output_dir="doda_darija")
```

### 2. Finetuner (GPU T4 requis)
```bash
from training.finetune import finetune
finetune(
    data_path   = "doda_darija",
    model_dir   = "/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/",
    output_path = "outputs",
    epochs      = 5,
)
```

### 3. Générer un audio
```bash
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
```bash
from interface.gradio_app import launch_interface
launch_interface(model, config, share=True)
```

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



###  Exécuter le modèle sur colab

### Important
Le repository GitHub **ne contient pas les fichiers du modèle entraîné** car ils sont trop volumineux.
Ils sont hébergés sur HuggingFace : [chaimaehde/xtts-darija](https://huggingface.co/chaimaehde/xtts-darija)

| Fichier | Taille | Description |
|---------|--------|-------------|
| `best_model_1370.pth` | ~1.5 GB | Poids du modèle GPT finetuné |
| `config.json` | ~5 KB | Configuration XTTS-v2 |
| `vocab.json` | ~1 KB | Vocabulaire du tokenizer arabe |

Ces 3 fichiers sont **téléchargés automatiquement** au premier lancement de `app.py`.

### Lancement sur Colab (GPU T4 requis)

```python
# Cellule 1 — Installation
!pip install -q coqui-tts gradio huggingface_hub soundfile
# Cellule 3 — Cloner et lancer
!git clone https://github.com/ChaimaeHde/xtts-darija30minTest.git
%cd xtts-darija30minTest
!python app.py
```

## Pourquoi aucun fine-tuning n'est nécessaire
Le fine-tuning est effectué **une seule fois** pour générer les fichiers finaux :
- `best_model_1370.pth`
- `config.json`
- `vocab.json`

Ensuite, le modèle peut être réutilisé à l'infini.

La fonction `load_model_once()` dans `app.py` :
- télécharge automatiquement les 3 fichiers depuis HuggingFace
- charge la configuration XTTS-v2
- initialise le modèle GPT
- charge les poids entraînés (checkpoint `best_model_1370.pth`)
- charge le vocabulaire arabe (`vocab.json`)
- envoie le modèle sur GPU (si disponible) ou CPU

Lorsque vous exécutez `app.py`, le modèle est **restauré**, pas réentraîné.

---

## Ce dont vous avez besoin
Pour exécuter le projet sur une autre machine :

1. Le repository du code (`git clone`)
2. Les dépendances installées (`pip install -r requirements.txt`)
3. Un token HuggingFace ([huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))
4. Les fichiers du modèle — **téléchargés automatiquement** depuis [chaimaehde/xtts-darija](https://huggingface.co/chaimaehde/xtts-darija) :
   - `best_model_1370.pth` — poids du modèle finetuné
   - `config.json` — configuration XTTS-v2
   - `vocab.json` — vocabulaire du tokenizer arabe

---


## Auteurs
Projet de Fin d'Année — Synthèse Vocale Expressive pour le Darija Marocain -

Chaimae Haddouche - Loubna Haouach
