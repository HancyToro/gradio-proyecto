#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""app_colab.py — Limpieza de audio: WAV 16kHz anti-alias + ClearVoice (Colab-ready).

Auto-suficiente: solo necesita clearvoice, gradio, soundfile, soxr, pyloudnorm, numpy.
En Colab se sube a /content/app_colab.py y se ejecuta con share=True.
"""

import json, os, sys, time, traceback

import numpy as np
import soundfile as sf
import soxr
import pyloudnorm as pyln
import gradio as gr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = ["auto", "FRCRN_SE_16K", "MossFormerGAN_SE_16K", "MossFormer2_SE_48K"]


def decode_to_array(path):
    """Cualquier formato -> (n, c) float32 en [-1,1]. Prioridad: soundfile > librosa > ffmpeg."""
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return data, sr, "soundfile"
    except Exception:
        pass
    try:
        import librosa
        y, sr = librosa.load(path, sr=None, mono=False)
        y = np.atleast_2d(y)
        return y.T.astype(np.float32), int(sr), "librosa"
    except Exception:
        pass
    import subprocess, tempfile
    wav = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-acodec",
                    "pcm_f32le", "-ar", "48000", wav], check=True)
    data, sr = sf.read(wav, dtype="float32", always_2d=True)
    os.remove(wav)
    return data, sr, "ffmpeg"


def convert_sample_rate(x, sr_in, sr_out):
    """Re-muestreo polyphase (soxr HQ): aplica filtro paso-bajo ANTES de bajar la tasa
    => sin aliasing; toda la información por debajo de Nyquist se preserva."""
    if sr_in == sr_out:
        return x
    return soxr.resample(x, sr_in, sr_out, quality="HQ").astype(np.float32)


def measure_lufs(x, sr):
    try:
        meter = pyln.Meter(sr)
        sig = x[:, 0] if x.ndim > 1 else x
        l = meter.integrated_loudness(sig)
        return float(l) if np.isfinite(l) else -120.0
    except Exception:
        return 0.0


def gain_to_lufs(x, lufs_before, target_lufs):
    gain_db = target_lufs - lufs_before
    if abs(gain_db) < 0.1:
        return x, 0.0, False
    peak = float(np.max(np.abs(x)))
    ceiling_hit = False
    if peak > 0 and peak * (10 ** (gain_db / 20)) > 0.999:
        gain_db = 20 * np.log10(0.999 / peak)
        ceiling_hit = True
    return (x * (10 ** (gain_db / 20))).astype(np.float32), gain_db, ceiling_hit


def to_wav_int16(path, x, sr):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sf.write(path, x, sr, subtype="PCM_16")


def preprocess(in_path, out_path=None, target_sr=16000, mono=True,
               normalize_lufs=False, target_lufs=-16.0, manifest_path=None):
    """Etapa 1: decodificar -> mono -> anti-alias/resample -> LUFS opcional -> WAV PCM_16."""
    base = os.path.splitext(os.path.basename(in_path))[0]
    out_path = out_path or os.path.join(OUTPUT_DIR, f"{base}_16k.wav")

    raw, sr_native, decoder = decode_to_array(in_path)
    info_in = {"formato_entrada": os.path.splitext(in_path)[1].lower(),
               "decoder_usado": decoder,
               "sample_rate_nativo_hz": sr_native,
               "canales_nativos": int(raw.shape[1]),
               "duracion_s": round(float(len(raw) / sr_native), 3)}

    x = raw
    if mono and x.shape[1] > 1:
        x = x.mean(axis=1, keepdims=True).astype(np.float32)

    lufs_before = measure_lufs(x, sr_native)
    x16 = convert_sample_rate(x, sr_native, target_sr)
    if x16.ndim == 1:
        x16 = x16[:, None]

    gain_db, ceiling_hit = 0.0, False
    if normalize_lufs:
        x16, gain_db, ceiling_hit = gain_to_lufs(x16, lufs_before, target_lufs)

    to_wav_int16(out_path, x16, target_sr)
    lufs_after = measure_lufs(x16, target_sr)

    manifest = {
        "archivo_entrada": in_path,
        "archivo_intermedio": out_path,
        "etapa": "normalizacion_a_wav_16khz",
        "entrada": info_in,
        "proceso": {"mono": mono, "target_sr_hz": target_sr,
                    "remuestreo": "soxr polyphase HQ (anti-alias)",
                    "normalizacion_lufs": bool(normalize_lufs),
                    "ganancia_aplicada_db": round(gain_db, 2),
                    "techo_alcanzado": ceiling_hit},
        "salida": {"sample_rate_hz": target_sr, "canales": 1,
                   "profundidad_bits": 16,
                   "lufs_antes": round(lufs_before, 2),
                   "lufs_despues": round(lufs_after, 2)},
    }
    if manifest_path:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


_CACHE = {"key": None, "net": None}


def _chunked_crossfade(net, x, sr, chunk_s=8.0, overlap_s=0.5):
    """Inferencia por chunks con cross-fade: evita OOM (CPU/Colab básico) y el bug de
    decode_one_audio_batch en clearvoice 0.1.2."""
    import torch
    n = len(x)
    chunk = max(int(sr * chunk_s), sr)
    ov = min(int(sr * overlap_s), chunk // 4)
    out = np.zeros(n); wsum = np.zeros(n)
    pos = 0
    while pos < n:
        end = min(pos + chunk, n)
        seg = x[pos:end]
        pad = sr // 100
        seg_p = np.concatenate([seg, np.zeros(pad, dtype=np.float32)])
        inp = torch.from_numpy(seg_p[None, :]).to(net.device)
        with torch.no_grad():
            if hasattr(net.model, "inference"):
                o = net.model.inference(inp)
            else:
                o = net.model(inp)
            if isinstance(o, (list, tuple)):
                o = o[0]
            y = o.detach().cpu().numpy().reshape(-1)[: len(seg_p)][: len(seg)].astype(np.float32)
        m = np.ones(len(seg))
        if pos > 0:
            m[:ov] *= np.linspace(0.0, 1.0, ov)
        if end < n:
            m[-ov:] *= np.linspace(1.0, 0.0, ov)
        out[pos:end] += y * m
        wsum[pos:end] += m
        pos = end - ov if end < n else n
    wsum[wsum == 0] = 1.0
    return (out / wsum).astype(np.float32)


def get_model(model_name, sr):
    from clearvoice import ClearVoice
    if model_name == "auto":
        model_name = "FRCRN_SE_16K" if sr <= 16000 else "MossFormer2_SE_48K"
    if _CACHE["key"] != model_name:
        cv = ClearVoice(task="speech_enhancement", model_names=[model_name])
        _CACHE["key"] = model_name
        _CACHE["net"] = cv.models[0]
    return model_name, _CACHE["net"]


def run_pipeline(audio_path, target_sr, normalize_lufs, target_lufs, model_choice, skip_denoise):
    if not audio_path:
        raise gr.Error("Sube un archivo de audio primero (mp3/wav/m4a/ogg/flac/webm...).")
    t0 = time.time()
    base = os.path.splitext(os.path.basename(audio_path))[0]
    stage1_out = os.path.join(OUTPUT_DIR, f"{base}_16k.wav")

    man1 = preprocess(audio_path, out_path=stage1_out, target_sr=int(target_sr),
                      mono=True, normalize_lufs=bool(normalize_lufs),
                      target_lufs=float(target_lufs))

    wav_16k = stage1_out
    denoised, man2 = None, None
    if not skip_denoise:
        x, sr_in = sf.read(wav_16k, dtype="float32", always_2d=True)
        chosen, net = get_model(model_choice, sr_in)
        native_sr = 16000 if chosen.endswith("_16K") else 48000
        feed = convert_sample_rate(x, sr_in, native_sr)
        y = _chunked_crossfade(net, feed[:, 0].astype(np.float32), native_sr)
        y = y[: len(feed)]
        y_out = convert_sample_rate(y[:, None], native_sr, sr_in)
        peak = float(np.max(np.abs(y_out))) + 1e-12
        if peak > 0.999:
            y_out = (y_out * (0.999 / peak)).astype(np.float32)
        denoised = os.path.join(OUTPUT_DIR, f"{base}_clean.wav")
        to_wav_int16(denoised, y_out, sr_in)
        man2 = {"etapa": "reduccion_ruido_clearvoice", "modelo": chosen,
                "sr_entrada_hz": int(sr_in), "sr_nativo_modelo_hz": int(native_sr),
                "lufs_entrada": round(measure_lufs(x, sr_in), 2),
                "lufs_salida": round(measure_lufs(y_out, sr_in), 2),
                "salidas": {"enhanced_same_sr": denoised}}

    elapsed = time.time() - t0
    combined = dict(man1)
    if man2:
        combined["denoise"] = man2
    combined["tiempo_total_s"] = round(elapsed, 2)
    status = (f"✅ Listo en {elapsed:.1f}s | Entrada: {man1['entrada']['sample_rate_nativo_hz']} Hz "
              f"{man1['entrada']['canales_nativos']} ch -> Salida: WAV {man1['salida']['sample_rate_hz']} Hz mono 16-bit"
              + (f" | Modelo: {man2['modelo']}" if man2 else " | (sin denoise)"))
    return wav_16k, denoised, status, combined


CSS = ".footer-note {text-align:center; font-size:0.85em; color:#888;}"


def build_ui(default_model="auto"):
    with gr.Blocks(title="Limpieza de Audio IA — WAV 16kHz + ClearVoice") as demo:
        gr.Markdown(
            "# 🎙️ Limpieza de Audio con IA\n"
            "**Pipeline:** ① Cualquier audio → **WAV PCM 16-bit** → **16 kHz sin aliasing** "
            "(filtro paso-bajo antes de re-muestrear, Nyquist-Shannon) → ② **Reducción de ruido "
            "ClearVoice** (ClearerVoice-Studio).\n")
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
                cb_skip = gr.Checkbox(value=False, label="Solo etapa 1 (saltar reducción de ruido)")
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
        gr.Markdown("<div class='footer-note'>Motor: ClearerVoice-Studio (clearvoice) · "
                    "Re-muestreo polyphase soxr</div>")
    return demo


if __name__ == "__main__":
    try:
        demo = build_ui()
        # share=True => enlace público https://xxxx.gradio.live (necesario en Colab)
        demo.queue().launch(theme=gr.themes.Soft(), css=CSS, show_error=True,
                            server_name="0.0.0.0", server_port=7860, share=True)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
