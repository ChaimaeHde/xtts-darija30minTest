import os
import numpy as np
import soundfile as sf
from jiwer import wer, cer
from faster_whisper import WhisperModel


def evaluate_wer_cer(audio_files, original_texts, whisper_model_size="large-v2", device="cuda"):
    """
    Calcule WER et CER sur une liste d\'audios via Whisper.

    Paramètres :
        audio_files    : liste de chemins vers les fichiers WAV
        original_texts : liste de textes originaux correspondants
        whisper_model_size : taille du modèle Whisper (large-v2 recommandé)

    Usage :
        from evaluation.evaluate import evaluate_wer_cer
        results, avg_wer, avg_cer = evaluate_wer_cer(
            audio_files    = ["audio1.wav", "audio2.wav"],
            original_texts = ["مرحبا كيف داير", "واش مزيان"]
        )

    Note : Le WER peut être élevé car Whisper est entraîné sur l\'arabe
    standard (MSA) et non sur le Darija dialectal. Le CER est plus
    représentatif de la qualité phonétique réelle.
    """
    print(f"Chargement Whisper {whisper_model_size}...")
    compute_type = "float16" if device == "cuda" else "int8"
    asr = WhisperModel(whisper_model_size, device=device, compute_type=compute_type)
    print("Whisper chargé")

    results = []
    for i, (audio_path, original) in enumerate(zip(audio_files, original_texts)):
        segments, _ = asr.transcribe(audio_path, language="ar")
        recognized  = " ".join([s.text for s in segments]).strip()

        score_wer = wer(original, recognized)
        score_cer = cer(original, recognized)

        results.append({
            "audio"      : os.path.basename(audio_path),
            "original"   : original,
            "recognized" : recognized,
            "WER"        : score_wer,
            "CER"        : score_cer,
        })

        print(f"--- Audio {i+1} ---")
        print(f"Original  : {original}")
        print(f"Reconnu   : {recognized}")
        print(f"WER       : {score_wer:.2%}")
        print(f"CER       : {score_cer:.2%}")

    avg_wer = np.mean([r["WER"] for r in results])
    avg_cer = np.mean([r["CER"] for r in results])

    print(f"\n{'='*50}")
    print(f"WER moyen : {avg_wer:.2%}")
    print(f"CER moyen : {avg_cer:.2%}")
    print(f"{'='*50}")

    return results, avg_wer, avg_cer


def compute_mos(notes_dict):
    """
    Calcule le MOS depuis un dictionnaire de notes humaines.

    Paramètres :
        notes_dict : dict avec critères comme clés et listes de notes comme valeurs
                     Notes de 1 à 5 par audio et par critère

    Usage :
        from evaluation.evaluate import compute_mos
        mos_global, mos_par_critere = compute_mos({
            "Intelligibilité" : [3, 4, 3, 4, 3, 4, 3, 4, 3, 4],
            "Naturalité"      : [3, 3, 2, 3, 3, 3, 2, 3, 3, 3],
            "Accent marocain" : [4, 4, 3, 4, 4, 4, 3, 4, 4, 4],
            "Expressivité"    : [2, 3, 2, 3, 2, 3, 2, 3, 2, 3],
        })

    Résultats obtenus dans notre expérimentation (10 audios, Darija 30 min) :
        MOS Global : 2.85 / 5
    """
    print("\nRÉSULTATS MOS")
    print("=" * 40)
    mos_par_critere = {}
    for critere, valeurs in notes_dict.items():
        mos = np.mean(valeurs)
        mos_par_critere[critere] = mos
        print(f"{critere:20} : {mos:.2f} / 5")

    mos_global = np.mean(list(mos_par_critere.values()))
    print("=" * 40)
    print(f"{'MOS Global':20} : {mos_global:.2f} / 5")

    return mos_global, mos_par_critere


def print_summary(avg_wer, avg_cer, mos_global):
    """
    Affiche un résumé interprétatif des métriques.

    Résultats de notre pilote (30 min DODa) :
        WER = 82.04% (élevé car Whisper/Darija incompatibles)
        CER = 38.49% (plus représentatif, phonétique correcte)
        MOS = 2.85/5 (acceptable pour 30 min de données)
    """
    print("\nRÉSUMÉ GLOBAL")
    print("=" * 50)
    print(f"WER moyen  : {avg_wer:.2%}")
    print(f"CER moyen  : {avg_cer:.2%}")
    print(f"MOS global : {mos_global:.2f} / 5")
    print("=" * 50)

    print("\nINTERPRÉTATION")
    if avg_wer < 0.20:
        print("WER : Très bon")
    elif avg_wer < 0.40:
        print("WER : Acceptable")
    else:
        print("WER : Élevé — normal si Whisper ne reconnaît pas le Darija dialectal")
        print("      → Utiliser le CER comme métrique principale")

    if avg_cer < 0.20:
        print("CER : Très bon — phonétique correcte")
    elif avg_cer < 0.40:
        print("CER : Acceptable pour un pilote de 30 min")
    else:
        print("CER : À améliorer — augmenter le corpus")

    if mos_global >= 4.0:
        print("MOS : Très bon")
    elif mos_global >= 3.0:
        print("MOS : Acceptable pour un pilote")
    else:
        print("MOS : Limité — expressivité insuffisante, augmenter les données")
