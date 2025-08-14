# app.py VERSI PERBAIKAN FINAL (Revisi 2)
import argparse
import contextlib
import io
import random
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from dia.model import Dia

# --- Variabel Global yang Aman untuk Diimpor ---
# Variabel-variabel ini sekarang berada di level atas, sehingga bisa diimpor.

css = """
#col-container {max-width: 90%; margin-left: auto; margin-right: auto;}
"""

default_text = "[S1] Dia is an open weights text to dialogue model. \n[S2] You get full control over scripts and voices. \n[S1] Wow. Amazing. (laughs) \n[S2] Try it now on Git hub or Hugging Face."
example_txt_path = Path("./example.txt")
if example_txt_path.exists():
    try:
        read_text = example_txt_path.read_text(encoding="utf-8").strip()
        if read_text:
            default_text = read_text
    except Exception as e:
        print(f"Warning: Could not read example.txt: {e}")

example_prompt_path = "./example_prompt.mp3"
examples_list = [
    [
        "[S1] Oh fire! Oh my goodness! What's the procedure? What to we do people? The smoke could be coming through an air duct! \n[S2] Oh my god! Okay.. it's happening. Everybody stay calm! \n[S1] What's the procedure... \n[S2] Everybody stay fucking calm!!!... Everybody fucking calm down!!!!! \n[S1] No! No! If you touch the handle, if its hot there might be a fire down the hallway! ",
        None, 3072, 3.0, 1.8, 0.95, 45, 1.0, -1,
    ],
    [
        "[S1] Open weights text to dialogue model. \n[S2] You get full control over scripts and voices. \n[S1] I'm biased, but I think we clearly won. \n[S2] Hard to disagree. (laughs) \n[S1] Thanks for listening to this demo. \n[S2] Try it now on Git hub and Hugging Face. \n[S1] If you liked our model, please give us a star and share to your friends. \n[S2] This was Nari Labs.",
        example_prompt_path if Path(example_prompt_path).exists() else None, 3072, 3.0, 1.8, 0.95, 45, 1.0, -1,
    ],
]


# --- Fungsi dan Definisi yang Aman untuk Diimpor ---

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def run_inference(
    # ... (isi fungsi run_inference tidak berubah) ...
    text_input: str, audio_prompt_text_input: str, audio_prompt_input: Optional[Tuple[int, np.ndarray]],
    max_new_tokens: int, cfg_scale: float, temperature: float, top_p: float,
    cfg_filter_top_k: int, speed_factor: float, seed: Optional[int] = None,
    model: Dia = None, device: torch.device = None,
):
    if model is None or device is None:
        raise gr.Error("Model and device must be provided to run_inference.")
    
    # --- (Salin-tempel seluruh isi fungsi run_inference asli Anda di sini) ---
    # Contoh sederhana:
    print(f"Running inference for text: {text_input[:30]}...")
    set_seed(seed if seed is not None and seed > 0 else random.randint(0, 2**32 - 1))
    with torch.inference_mode():
        output_audio_np = model.generate(text_input, max_tokens=max_new_tokens, cfg_scale=cfg_scale, temperature=temperature, top_p=top_p, use_torch_compile=False)
    output_sr = 44100
    output_audio = (output_sr, output_audio_np)
    return output_audio, seed, "Inference complete."


# --- Fungsi Wrapper untuk Notebook ---
def start_app_and_get_url():
    print("🚀 Memulai proses dari dalam start_app_and_get_url...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = "float16"
    else:
        device = torch.device("cpu")
        dtype = "float32"
    print(f"Menggunakan device: {device} dengan dtype: {dtype}")

    model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype=dtype, device=device)

    # UI Gradio sekarang menggunakan variabel global yang sudah kita definisikan di atas
    with gr.Blocks(css=css, theme="gradio/dark") as demo:
        # ... (Kode UI Anda di sini, tidak perlu diubah) ...
        gr.Markdown("# Nari Text-to-Speech Synthesis")
        with gr.Row():
            text_input = gr.Textbox(label="Text To Generate", value=default_text, lines=5)
            run_button = gr.Button("Generate Audio")
        audio_output = gr.Audio(label="Generated Audio")
        # ... (Lanjutkan membangun UI lengkap Anda di sini)

        from functools import partial
        inference_with_model = partial(run_inference, model=model, device=device)
        
        # Sederhanakan input untuk contoh, Anda harus melengkapinya sesuai UI
        run_button.click(
            fn=inference_with_model,
            inputs=[text_input, gr.State(None), gr.State(None), gr.State(1024), gr.State(3.0), gr.State(1.8), gr.State(0.95), gr.State(45), gr.State(1.0), gr.State(-1)],
            outputs=[audio_output, gr.Textbox(), gr.Textbox()]
        )

    print("📡 Meluncurkan Gradio...")
    _, _, public_url = demo.launch(share=True)
    return public_url


# --- Blok Eksekusi Utama (HANYA berjalan jika file dijalankan langsung) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradio interface for Nari TTS")
    parser.add_argument("--device", type=str, default=None, help="Force device (e.g., 'cuda', 'mps', 'cpu')")
    parser.add_argument("--share", action="store_true", help="Enable Gradio sharing")
    args, unknown = parser.parse_known_args()
    
    public_url = start_app_and_get_url()
    print(f"✅ Aplikasi berjalan. URL Publik: {public_url}")
    
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("🔌 Server dihentikan.")
