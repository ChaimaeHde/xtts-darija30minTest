import gradio as gr
import soundfile as sf
import numpy as np


def launch_interface(model, config, share=True):
    """
    Lance l\'interface Gradio pour tester le modèle TTS Darija.

    Paramètres :
        model  : modèle chargé via load_model()
        config : config chargée via load_model()
        share  : True pour générer un lien public Gradio

    Usage :
        from inference.generate import load_model
        from interface.gradio_app import launch_interface

        model, config = load_model(
            checkpoint_path = "/content/pfa_outputs/best_model_1370.pth",
            config_path     = "/content/pfa_outputs/config.json",
            vocab_path      = "/root/.local/share/tts/.../vocab.json",
        )
        launch_interface(model, config, share=True)
    """

    def generate_darija_tts(text, ref_audio):
        if not text or not text.strip():
            return None, "Texte vide"
        if ref_audio is None:
            return None, "Pas d\'audio de référence"
        try:
            sr, audio_data = ref_audio

            # Conversion float32 si nécessaire
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
                if audio_data.max() > 1.0:
                    audio_data = audio_data / 32768.0

            # Sauvegarde audio de référence
            sf.write("/tmp/ref_input.wav", audio_data, sr)

            # Génération
            outputs = model.synthesize(
                text        = text,
                config      = config,
                speaker_wav = "/tmp/ref_input.wav",
                language    = "ar",
            )

            out_path = "/tmp/output_darija.wav"
            sf.write(out_path, outputs["wav"], 24000)
            return out_path, "Audio généré avec succès"

        except Exception as e:
            return None, f"Erreur : {str(e)}"

    with gr.Blocks(title="TTS Darija Marocain") as demo:
        gr.Markdown("# TTS Darija Marocain")
        gr.Markdown(
            "Synthèse vocale pour le dialecte marocain "
            "— XTTS-v2 finetuné sur DODa (30 min)"
        )

        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label       = "Texte en Darija",
                    placeholder = "مرحبا، كيف داير؟",
                    lines       = 3,
                )
                ref_audio_input = gr.Audio(
                    label   = "Audio de référence (voix à cloner)",
                    type    = "numpy",
                    sources = ["upload", "microphone"],
                )
                btn = gr.Button("Générer", variant="primary")

            with gr.Column():
                audio_output  = gr.Audio(label="Voix générée", type="filepath")
                status_output = gr.Textbox(label="Statut")

        btn.click(
            fn      = generate_darija_tts,
            inputs  = [text_input, ref_audio_input],
            outputs = [audio_output, status_output],
        )

        gr.Examples(
            examples = [
                ["مرحبا، كيف داير؟ واش كلشي مزيان؟"],
                ["واش نتا مزيان؟ شنو كاين الجديد اليوم؟"],
                ["الجو مزيان بزاف اليوم، خرجنا نتفرجو"],
                ["ما فهمتش"],
                ["الله يحفظك، بارك الله فيك"],
            ],
            inputs = [text_input],
        )

    demo.launch(share=share, debug=False)
