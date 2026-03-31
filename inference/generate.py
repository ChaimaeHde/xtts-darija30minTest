import os
import torch
import soundfile as sf
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


def load_model(checkpoint_path, config_path, vocab_path):
    """
    Charge le modèle finetuné pour la génération.

    Paramètres :
        checkpoint_path : chemin vers best_model_1370.pth
        config_path     : chemin vers config.json
        vocab_path      : chemin vers vocab.json

    Usage :
        from inference.generate import load_model
        model, config = load_model(
            checkpoint_path = "/content/pfa_outputs/best_model_1370.pth",
            config_path     = "/content/pfa_outputs/config.json",
            vocab_path      = "/root/.local/share/tts/.../vocab.json",
        )
    """
    config = XttsConfig()
    config.load_json(config_path)

    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_path = checkpoint_path,
        vocab_path      = vocab_path,
        eval            = True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    print(f"✅ Modèle chargé sur {device}")
    return model, config


def generate(model, config, text, speaker_wav, language="ar", output_path="output.wav"):
    """
    Génère un audio depuis un texte et un audio de référence (voice cloning).

    Paramètres :
        model       : modèle chargé via load_model()
        config      : config chargée via load_model()
        text        : texte en Darija à synthétiser
        speaker_wav : chemin vers l\'audio de référence (voix à cloner)
        language    : code langue (ar pour Darija)
        output_path : chemin de sauvegarde de l\'audio généré

    Usage :
        from inference.generate import generate
        generate(
            model       = model,
            config      = config,
            text        = "مرحبا، كيف داير؟",
            speaker_wav = "/content/ref.wav",
            output_path = "/content/output.wav",
        )
    """
    outputs = model.synthesize(
        text        = text,
        config      = config,
        speaker_wav = speaker_wav,
        language    = language,
    )
    sf.write(output_path, outputs["wav"], 24000)
    print(f"✅ Audio généré : {output_path}")
    return output_path


def generate_batch(model, config, texts, speaker_wav, output_dir="generated", language="ar"):
    """
    Génère plusieurs audios en batch.

    Paramètres :
        texts      : liste de textes à synthétiser
        speaker_wav: audio de référence unique pour tous les audios
        output_dir : dossier de sauvegarde

    Usage :
        from inference.generate import generate_batch
        generate_batch(
            model       = model,
            config      = config,
            texts       = ["مرحبا", "كيف داير؟", "واش مزيان؟"],
            speaker_wav = "/content/ref.wav",
            output_dir  = "/content/generated",
        )
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, text in enumerate(texts):
        out = os.path.join(output_dir, f"audio_{i:03d}.wav")
        generate(model, config, text, speaker_wav,
                 language=language, output_path=out)
        paths.append(out)
    print(f"✅ {len(paths)} audios générés dans {output_dir}/")
    return paths
