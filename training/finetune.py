import os
import torch
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.utils.manage import ModelManager
from trainer import Trainer, TrainerArgs


def download_base_model(model_dir):
    """Télécharge dvae.pth et mel_stats.pth si manquants."""
    files = [
        "https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth",
        "https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth",
    ]
    for url in files:
        fname = os.path.basename(url)
        dest  = os.path.join(model_dir, fname)
        if not os.path.isfile(dest):
            print(f"Téléchargement {fname}...")
            ModelManager._download_model_files([url], model_dir, progress_bar=True)
        else:
            print(f" {fname} déjà présent")


def finetune(
    data_path,
    model_dir,
    output_path,
    language   = "ar",
    epochs     = 5,
    batch_size = 2,
    lr         = 5e-6,
    grad_accum = 126,
):
    """
    Lance le finetuning XTTS-v2 sur un dataset Darija.

    Paramètres :
        data_path   : dossier contenant wavs/ et train.csv
        model_dir   : dossier du modèle XTTS-v2 de base
        output_path : dossier de sauvegarde des checkpoints
        language    : code langue (ar = arabe/Darija)
        epochs      : nombre d\'epochs (5 pour 30 min de data)
        batch_size  : taille du batch (2 pour GPU T4)
        lr          : learning rate (5e-6 recommandé pour fine-tuning)
        grad_accum  : gradient accumulation steps (126 pour batch effectif 252)

    Usage :
        from training.finetune import finetune
        finetune(
            data_path   = "/content/doda_darija",
            model_dir   = "/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/",
            output_path = "/content/outputs",
            epochs      = 5,
        )
    """
    os.makedirs(output_path, exist_ok=True)

    # Télécharger les fichiers manquants du modèle de base
    download_base_model(model_dir)

    # Configuration du dataset
    dataset_config = BaseDatasetConfig(
        formatter        = "ljspeech",
        dataset_name     = os.path.basename(data_path),
        path             = data_path,
        meta_file_train  = "train.csv",
        meta_file_val    = "",
        ignored_speakers = None,
        language         = language,
    )

    # Configuration du modèle GPT
    model_args = GPTArgs(
        max_conditioning_length            = 132300,
        min_conditioning_length            = 66150,
        debug_loading_failures             = False,
        max_wav_length                     = 255995,
        max_text_length                    = 200,
        mel_norm_file                      = os.path.join(model_dir, "mel_stats.pth"),
        dvae_checkpoint                    = os.path.join(model_dir, "dvae.pth"),
        xtts_checkpoint                    = os.path.join(model_dir, "model.pth"),
        tokenizer_file                     = os.path.join(model_dir, "vocab.json"),
        gpt_num_audio_tokens               = 1026,
        gpt_start_audio_token              = 1024,
        gpt_stop_audio_token               = 1025,
        gpt_use_masking_gt_prompt_approach = True,
        gpt_use_perceiver_resampler        = True,
    )

    # Configuration audio
    audio_config = XttsAudioConfig(
        sample_rate        = 22050,
        dvae_sample_rate   = 22050,
        output_sample_rate = 24000,
    )

    # Configuration de l\'entraînement
    config = GPTTrainerConfig(
        epochs                       = epochs,
        output_path                  = output_path,
        model_args                   = model_args,
        run_name                     = f"xtts_{os.path.basename(data_path)}",
        project_name                 = "xtts_darija",
        dashboard_logger             = "tensorboard",
        logger_uri                   = None,
        audio                        = audio_config,
        batch_size                   = batch_size,
        batch_group_size             = 48,
        eval_batch_size              = batch_size,
        num_loader_workers           = 2,
        eval_split_max_size          = 256,
        print_step                   = 50,
        plot_step                    = 100,
        log_model_step               = 100,
        save_step                    = 1000,
        save_n_checkpoints           = 1,
        save_checkpoints             = True,
        print_eval                   = False,
        optimizer                    = "AdamW",
        optimizer_wd_only_on_weights = True,
        optimizer_params             = {"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr                           = lr,
        lr_scheduler                 = "MultiStepLR",
        lr_scheduler_params          = {"milestones": [50000, 150000, 300000], "gamma": 0.5, "last_epoch": -1},
        test_sentences               = [],
        datasets                     = [dataset_config],
    )

    # Chargement des données
    train_samples, eval_samples = load_tts_samples(
        [dataset_config],
        eval_split          = True,
        eval_split_max_size = 256,
        eval_split_size     = 0.1,
    )
    print(f" Train : {len(train_samples)} samples")
    print(f" Eval  : {len(eval_samples)} samples")

    # Initialisation du modèle et lancement
    model = GPTTrainer.init_from_config(config)
    trainer = Trainer(
        TrainerArgs(
            restore_path     = None,
            skip_train_epoch = False,
            start_with_eval  = False,
            grad_accum_steps = grad_accum,
        ),
        config,
        output_path   = output_path,
        model         = model,
        train_samples = train_samples,
        eval_samples  = eval_samples,
    )

    print(f"Démarrage du finetuning — {epochs} epochs")
    trainer.fit()
    print(f" Modèle sauvegardé dans {output_path}")
    return output_path
