"""
Pipeline complet de préparation du dataset DODa pour XTTS-v2.
Reproduit exactement les cellules 3 à 7 du notebook Colab.
"""

import os, re, subprocess
import pandas as pd
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

TEXT_COL = "darija_Arab_new"

def download_doda(n_samples=650, output_dir="/content/doda_f1_30min"):
    """Cellule 3 — Télécharge DODa et sauvegarde les WAV."""
    ds = load_dataset("atlasia/DODa-audio-dataset", split="train")
    subset = ds.select(range(n_samples))
    os.makedirs(f"{output_dir}/wavs", exist_ok=True)
    rows = []
    for i, item in enumerate(tqdm(subset, desc="Saving WAVs")):
        fname = f"utt_{i:04d}.wav"
        sf.write(f"{output_dir}/wavs/{fname}", item["audio"]["array"], item["audio"]["sampling_rate"])
        rows.append({"file_name": fname, "text": item.get(TEXT_COL, "")})
    pd.DataFrame(rows).to_csv(f"{output_dir}/metadata.csv", index=False)
    print(f"✅ {len(rows)} samples sauvegardés")

def clean_dataset(output_dir="/content/doda_f1_30min"):
    """Cellule 4 — Filtre et déduplique."""
    df = pd.read_csv(f"{output_dir}/metadata.csv")
    df["text_len"] = df["text"].str.len()
    df = df[(df["text_len"] > 5) & (df["text_len"] < 150)]
    df = df.drop_duplicates(subset=["text"])
    df.to_csv(f"{output_dir}/metadata_clean.csv", index=False)
    print(f"✅ Après nettoyage: {len(df)} samples")
    return df

def normalize_darija(text):
    """Cellule 5 — Normalisation du texte Darija."""
    text = re.sub(r"[^\w\s\u0600-\u06FF]", "", text)
    return " ".join(text.split())

def create_train_csv(output_dir="/content/doda_f1_30min"):
    """Cellule 6 — Crée train.csv au format LJSpeech."""
    df = pd.read_csv(f"{output_dir}/metadata_clean.csv")
    df["text_norm"] = df["text"].apply(normalize_darija)
    with open(f"{output_dir}/train.csv", "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(f"wavs/{row['file_name']}|{row['text_norm']}|{row['text_norm']}\n")
    # Fix id field (enlève "wavs/" et ".wav")
    df_csv = pd.read_csv(f"{output_dir}/train.csv", sep="|", header=None, names=["id","text1","text2"])
    df_csv["id"] = df_csv["id"].str.replace("wavs/","",regex=False).str.replace(".wav","",regex=False)
    df_csv.to_csv(f"{output_dir}/train.csv", sep="|", header=False, index=False)
    print(f"✅ train.csv créé — exemple id: {df_csv['id'].iloc[0]}")

def convert_to_22050(output_dir="/content/doda_f1_30min"):
    """Cellule 7 — Convertit les WAV à 22050 Hz (requis par XTTS-v2)."""
    wav_dir = f"{output_dir}/wavs"
    wavs = [f for f in os.listdir(wav_dir) if f.endswith(".wav")]
    if sf.info(os.path.join(wav_dir, wavs[0])).samplerate == 22050:
        print("✅ WAV déjà à 22050 Hz")
        return
    for fname in tqdm(wavs, desc="Converting"):
        src = os.path.join(wav_dir, fname)
        tmp = src.replace(".wav", "_tmp.wav")
        ret = subprocess.run(["ffmpeg","-y","-i",src,"-ar","22050","-ac","1","-sample_fmt","s16",tmp], capture_output=True)
        if ret.returncode == 0:
            os.replace(tmp, src)
    print("✅ Conversion 22050 Hz terminée")

def prepare_all(n_samples=650, output_dir="/content/doda_f1_30min"):
    """Lance tout le pipeline dans l'ordre."""
    download_doda(n_samples, output_dir)
    clean_dataset(output_dir)
    create_train_csv(output_dir)
    convert_to_22050(output_dir)
    print("✅ Dataset prêt pour le finetuning !")
