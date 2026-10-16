#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py — Interfaz Gradio del pipeline de limpieza de audio

Pipeline (2 etapas):
  1. Normalización: cualquier formato subido -> WAV PCM 16-bit @ 16 kHz,
     con filtro paso-bajo antialiasing ANTES de re-muestrear (Nyquist-Shannon)
     y normalización de loudness BS.1770-4 opcional (scripts/audio_preprocess.py).
  2. Reducción de ruido: ClearVoice (ClearerVoice-Studio) — FRCRN_SE_16K por
     defecto; MossFormer2_SE_48K / MossFormerGAN_SE_16K opcionales
     (scripts/denoise_clearvoice.py).

Uso:
  python3 app.py [--share] [--port 7860] [--model auto|FRCRN_SE_16K|MossFormer2_SE_48K|MossFormerGAN_SE_16K]
"""

import argparse
import json
import os
import sys
import time
import traceback

import gradio as gr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))

from audio_preprocess import preprocess                      # noqa: E402
from denoise_clearvoice import enhance, _infer_chunked       # noqa: E402

MODELS = ["auto", "FRCRN_SE_16K", "MossFormer2_SE_48K", "MossFormerGAN_SE_16K"]

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------- modelo singleton
# Cargar el modelo es lento (~segundos) y consume RAM: se cachea en memoria.
_CACHE = {"key": None, "net": None}


def get_model(model_name: str, sr: int):
    """Devuelve (nombre_resuelto, red ClearVoice), recargando solo si cambia el modelo."""
    from clearvoice import ClearVoice
    if model_name == "auto":
        model_name = "FRCRN_SE_16K" if sr <= 16000 else "MossFormer2_SE_48K"
    if _CACHE["key"] != model_name:
        cv = ClearVoice(task="speech_enhancement", model_names=[model_name])
        _CACHE["key"] = model_name
        _CACHE["net"] = cv.models[0]
    return model_name, _CACHE["net"]


# --------------------------------------------------------------------------- lógica del pipeline
def run_pipeline(audio_path: str,
                 target_sr: int,
                 normalize_lufs: bool,
                 target_lufs: float,
                 model_choice: str,
                 skip_denoise: bool):
    """Ejecuta etapa 1 (+ etapa 2) y devuelve los componentes de la UI."""
    if not audio_path:
        raise gr.Error("Sube un archivo de audio primero (mp3/wav/m4a/ogg/flac/webm...).")

    t0 = time.time()
    base = os.path.splitext(os.path.basename(audio_path))[0]
    stage1_out = os.path.join(OUTPUT_DIR, f"{base}_16k.wav")

    # ---------------- Etapa 1: WAV + re-muestreo anti-aliasing ----------------
    man1 = preprocess(audio_path,
                      out_path=stage1_out,
                      target_sr=int(target_sr),
                      mono=True,
                      normalize_lufs=bool(normalize_lufs),
                      target_lufs=float(target_lufs))

    wav_16k = stage1_out
    denoised = None
    man2 = None

    if not skip_denoise:
        # ---------------- Etapa 2: ClearVoice (con modelo cacheado) -----------
        import numpy as np
        import soundfile as sf
        from audio_preprocess import to_wav_int16, convert_sample_rate, measure_lufs

        x, sr_in = sf.read(wav_16k, dtype="float32", always_2d=True)
        chosen, net = get_model(model_choice, sr_in)
        native_sr = 16000 if chosen.endswith("_16K") else 48000

        feed = convert_sample_rate(x, sr_in, native_sr)
        y = _infer_chunked(net, feed[:, 0].astype(np.float32), native_sr)
        y = y[: len(feed)]
        y_out = convert_sample_rate(y[:, None], native_sr, sr_in)
        peak = float(np.max(np.abs(y_out))) + 1e-12
        if peak > 0.999:
            y_out = (y_out * (0.999 / peak)).astype(np.float32)

        denoised = os.path.join(OUTPUT_DIR, f"{base}_clean.wav")
        to_wav_int16(denoised, y_out, sr_in)
        man2 = {
            "etapa": "reduccion_ruido_clearvoice",
            "modelo": chosen,
            "sr_entrada_hz": int(sr_in),
            "sr_nativo_modelo_hz": int(native_sr),
            "lufs_entrada": round(measure_lufs(x, sr_in), 2),
            "lufs_salida": round(measure_lufs(y_out, sr_in), 2),
            "salidas": {"enhanced_same_sr": denoised},
        }

    elapsed = time.time() - t0
    combined = dict(man1)
    if man2:
        combined["denoise"] = man2
    combined["tiempo_total_s"] = round(elapsed, 2)

    pretty = json.dumps(combined, ensure_ascii=False, indent=2)
    status = (f"✅ Listo en {elapsed:.1f}s | Entrada: {man1['entrada']['sample_rate_nativo_hz']} Hz "
              f"{man1['entrada']['canales_nativos']} ch -> Salida: WAV {man1['salida']['sample_rate_hz']} Hz mono 16-bit"
              + (f" | Modelo: {man2['modelo']}" if man2 else " | (sin denoise)"))
    return wav_16k, denoised, status, pretty


def run_from_file(filepath: str, sample_rate: int):
    """Variante para procesar audios grabados con el micrófono de Gradio."""
    return run_pipeline(filepath, sample_rate, True, -16.0, "auto", False)


# --------------------------------------------------------------------------- UI
CSS = """
.footer-note {text-align:center; font-size:0.85em; color:#888;}
"""


def build_ui(default_model: str = "auto"):
    with gr.Blocks(title="Limpieza de Audio IA — WAV 16kHz + ClearVoice") as demo:
        gr.Markdown(
            "# 🎙️ Limpieza de Audio con IA\n"
            "**Pipeline:** ① Cualquier audio → **WAV PCM 16-bit** → **16 kHz sin aliasing** "
            "(filtro paso-bajo antes de re-muestrear, Nyquist-Shannon) → ② **Reducción de ruido "
            "ClearVoice** (ClearerVoice-Studio).\n"
        )

        with gr.Row():
            with gr.Column(scale=1):
                inp_audio = gr.Audio(label="1) Sube tu audio (mp3, wav, m4a, ogg, flac, webm...)",
                                     type="filepath", sources=["upload", "microphone"])
                sr_slider = gr.Slider(minimum=8000, maximum=48000, step=8000, value=16000,
                                      label="Sample rate objetivo (Hz)")
                cb_lufs = gr.Checkbox(value=True,
                                      label="Normalizar loudness a LUFS (BS.1770-4, ganancia lineal: no pierde información)")
                lufs_num = gr.Number(value=-16.0, label="LUFS objetivo", precision=1)
                dd_model = gr.Dropdown(choices=MODELS, value=default_model,
                                       label="Modelo ClearVoice (auto = FRCRN_SE_16K para ≤16 kHz)")
                cb_skip = gr.Checkbox(value=False,
                                      label="Solo etapa 1 (saltar reducción de ruido)")
                btn = gr.Button("🚀 Procesar", variant="primary")

            with gr.Column(scale=1):
                out_status = gr.Textbox(label="Estado", interactive=False)
                out_stage1 = gr.Audio(label="Etapa 1 — WAV normalizado (16 kHz, anti-alias)", type="filepath")
                out_stage2 = gr.Audio(label="Etapa 2 — Audio limpio (ClearVoice)", type="filepath")
                with gr.Accordion("📄 Manifiesto JSON (auditable)", open=False):
                    out_json = gr.Json(label="Métricas del proceso")

        btn.click(fn=run_pipeline,
                  inputs=[inp_audio, sr_slider, cb_lufs, lufs_num, dd_model, cb_skip],
                  outputs=[out_stage1, out_stage2, out_status, out_json])

        inp_audio.change(fn=lambda fp, sr: run_pipeline(fp, sr, True, -16.0, default_model, False)[0],
                         inputs=[inp_audio, sr_slider],
                         outputs=[out_stage1], show_progress="hidden")

        gr.Markdown(
            "<div class='footer-note'>Motor: ClearerVoice-Studio (clearvoice) · "
            "Re-muestreo polyphase soxr · Checkpoints locales en <code>checkpoints/</code></div>"
        )
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="Crear enlace público gradio.live")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--model", default="auto", choices=MODELS)
    args = ap.parse_args()

    demo = build_ui(default_model=args.model)
    demo.queue().launch(theme=gr.themes.Soft(), css=CSS, show_error=True,server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
