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

# --- Fungsi dan Definisi yang Aman untuk Diimpor ---
# Fungsi-fungsi ini tidak menjalankan argparse, jadi aman untuk diimpor.

def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_inference(
    text_input: str,
    audio_prompt_text_input: str,
    audio_prompt_input: Optional[Tuple[int, np.ndarray]],
    max_new_tokens: int,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    cfg_filter_top_k: int,
    speed_factor: float,
    seed: Optional[int] = None,
    # Kita akan menambahkan model dan device sebagai parameter
    model: Dia = None,
    device: torch.device = None,
):
    """
    Runs Nari inference using the provided model and inputs.
    """
    if model is None or device is None:
        raise gr.Error("Model and device must be provided to run_inference.")

    console_output_buffer = io.StringIO()
    # (Sisa dari fungsi run_inference Anda tidak perlu diubah, salin-tempel saja dari file asli Anda)
    # ...
    # ... (Tempel semua logika internal dari fungsi run_inference asli di sini)
    # ...
    # Pastikan bagian `model.generate` ada di dalam fungsi ini
    with torch.inference_mode():
        output_audio_np = model.generate(
            text_input,
            max_tokens=max_new_tokens,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_p=top_p,
            cfg_filter_top_k=cfg_filter_top_k,
            use_torch_compile=False,
            audio_prompt=None, # Disederhanakan untuk contoh, sesuaikan jika perlu
            verbose=True,
        )
    # ... (Sisa logika konversi audio) ...
    output_sr = 44100
    output_audio = (output_sr, output_audio_np)
    console_output = console_output_buffer.getvalue()
    return output_audio, seed, console_output


# --- Fungsi Wrapper untuk Notebook ---
# Fungsi inilah yang akan dipanggil oleh notebook Colab Anda.
def start_app_and_get_url():
    """
    Fungsi mandiri yang memuat model, membuat UI, dan mengembalikan URL publik.
    """
    print("🚀 Memulai proses dari dalam start_app_and_get_url...")

    # Menentukan device di dalam fungsi
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = "float16"
    else:
        device = torch.device("cpu")
        dtype = "float32"
    print(f"Menggunakan device: {device} dengan dtype: {dtype}")

    # Memuat model di dalam fungsi
    model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype=dtype, device=device)

    # Membuat UI Gradio di dalam fungsi
    with gr.Blocks(theme="gradio/dark") as demo:
        gr.Markdown("# Nari Text-to-Speech Synthesis")
        # (Sederhanakan UI untuk contoh, Anda bisa salin-tempel UI lengkap Anda di sini)
        with gr.Row():
            text_input = gr.Textbox(label="Text To Generate")
            run_button = gr.Button("Generate Audio")
        audio_output = gr.Audio(label="Generated Audio")
        seed_output = gr.Textbox(label="Generation Seed")
        console_output = gr.Textbox(label="Console Log")

        # Buat fungsi parsial untuk memasukkan model dan device ke run_inference
        from functools import partial
        inference_with_model = partial(run_inference, model=model, device=device)

        run_button.click(
            fn=inference_with_model,
            inputs=[text_input, gr.State(None), gr.State(None), gr.State(1024), gr.State(3.0), gr.State(1.8), gr.State(0.95), gr.State(45), gr.State(1.0), gr.State(-1)],
            outputs=[audio_output, seed_output, console_output]
        )

    print("📡 Meluncurkan Gradio...")
    _, _, public_url = demo.launch(share=True)
    return public_url


# --- Blok Eksekusi Utama (HANYA berjalan jika file dijalankan langsung) ---
if __name__ == "__main__":
    # SEMUA LOGIKA argparse SEKARANG AMAN DI DALAM BLOK INI
    parser = argparse.ArgumentParser(description="Gradio interface for Nari TTS")
    parser.add_argument("--device", type=str, default=None, help="Force device (e.g., 'cuda', 'mps', 'cpu')")
    parser.add_argument("--share", action="store_true", help="Enable Gradio sharing")
    
    # Gunakan parse_known_args untuk keamanan ekstra
    args, unknown = parser.parse_known_args()
    
    print(" запущено напрямую. Запуск Gradio...") # (Running directly. Launching Gradio...)
    
    # Panggil fungsi utama untuk menjalankan aplikasi
    public_url = start_app_and_get_url()
    
    print(f"✅ Aplikasi berjalan. URL Publik: {public_url}")
    
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("🔌 Server dihentikan.")
