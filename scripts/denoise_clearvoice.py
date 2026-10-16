#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
denoise_clearvoice.py — Etapa 2: reducción de ruido con ClearVoice (ClearerVoice-Studio)

Modelos soportados (task='speech_enhancement'):
  * FRCRN_SE_16K        -> trabaja internamente a 16 kHz. Ideal para audio ya normalizado
                           a 16 kHz (etapa 1). El más ligero: ~40 M params.
  * MossFormer2_SE_48K  -> modelo insignia, banda completa hasta 24 kHz (calidad máxima).

Estrategia "sin pérdida" implementada aquí:
  - El WAV 16 kHz de la etapa 1 se alimenta DIRECTAMENTE al modelo 16k (no hay resampleo
    adicional => cero artefactos extra).
  - Para el modelo 48k, el audio se sube a 48 kHz por interpolación polyphase (no inventa
    contenido sobre 8 kHz, pero preserva exactamente toda la banda 0-8 kHz que el modelo
    procesa en su resolución nativa) y el resultado se devuelve a la tasa original.
  - Se guarda además una versión del output a 16 kHz para downstream (ASR / Seed-VC).

Uso CLI:
  python3 scripts/denoise_clearvoice.py ENTRADA_WAV [-o SALIDA.wav] [--model auto|FRCRN_SE_16K|MossFormer2_SE_48K]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_preprocess import decode_to_array, to_wav_int16, convert_sample_rate, measure_lufs  # noqa: E402

MODEL_BY_SR = {16000: "FRCRN_SE_16K", 48000: "MossFormer2_SE_48K", 8000: "FRCRN_SE_16K"}


def _infer_chunked(net, x: np.ndarray, sr: int, chunk_s: float = 8.0, overlap_s: float = 0.5) -> np.ndarray:
    """Inferencia por segmentos con solapamiento + cross-fade lineal.

    Motivos de ingeniería:
      * Evita el bug de `decode_one_audio_batch` en clearvoice 0.1.2 (usa `np.zeros(b, t)`
        con args posicionales -> TypeError en la ruta segmentada y OOM en CPU con RAM baja).
      * Limita el pico de memoria (clave para CPU con RAM escasa, p.ej. Colab básico).
      * El cross-fade elimina discontinuidades audibles entre fragmentos.
    """
    import torch

    n = len(x)
    chunk = max(int(sr * chunk_s), sr)          # >= 1 s por fragmento
    ov = min(int(sr * overlap_s), chunk // 4)   # solape <= 25% del fragmento
    out = np.zeros(n, dtype=np.float64)
    wsum = np.zeros(n, dtype=np.float64)

    pos = 0
    while pos < n:
        end = min(pos + chunk, n)
        seg = x[pos:end]
        pad = sr // 100                        # colchón anti-frontera del modelo (10 ms)
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

        m = np.ones(len(seg), dtype=np.float64)
        if pos > 0:
            m[:ov] *= np.linspace(0.0, 1.0, ov)
        if end < n:
            m[-ov:] *= np.linspace(1.0, 0.0, ov)
        out[pos:end] += y * m
        wsum[pos:end] += m
        pos = end - ov if end < n else n

    wsum[wsum == 0] = 1.0
    return (out / wsum).astype(np.float32)


def pick_model(model: str, sr: int) -> str:
    if model and model != "auto":
        return model
    return MODEL_BY_SR.get(sr, "MossFormer2_SE_48K")


def enhance(in_path: str, out_path: str | None = None, model: str = "auto",
            also_16k: bool = True, device: str = "cpu") -> dict:
    from clearvoice import ClearVoice

    base = os.path.splitext(os.path.basename(in_path))[0]
    out_path = out_path or os.path.join("outputs", f"{base}_clean.wav")

    x, sr_in, decoder = decode_to_array(in_path)          # (n, c) float32
    if x.shape[1] > 1:
        x = x.mean(axis=1, keepdims=True).astype(np.float32)

    chosen = pick_model(model, sr_in)
    native_sr = 16000 if chosen.endswith("_16K") else 48000

    # Alimentamos al modelo en su tasa NATIVA mediante re-muestreo polyphase reversible
    # (banda útil preservada; ver docstring de convert_sample_rate).
    feed = convert_sample_rate(x, sr_in, native_sr)
    if feed.ndim == 1:
        feed = feed[:, None]

    cv = ClearVoice(task="speech_enhancement", model_names=[chosen])
    net = cv.models[0]
    y = _infer_chunked(net, feed[:, 0].astype(np.float32), native_sr)  # (T,) float32
    y = y[: len(feed)]              # recorte a longitud de entrada (garantía del wrapper)

    # Re-mapeamos el resultado a la tasa de trabajo original del pipeline (p.ej. 16 kHz)
    y_out = convert_sample_rate(y[:, None], native_sr, sr_in)
    peak = float(np.max(np.abs(y_out))) + 1e-12           # seguridad anti clip post-modelo
    if peak > 0.999:
        y_out = (y_out * (0.999 / peak)).astype(np.float32)

    to_wav_int16(out_path, y_out, sr_in)
    outs = {"enhanced_same_sr": out_path}

    if also_16k and sr_in != 16000:
        p16 = os.path.splitext(out_path)[0] + "_16k.wav"
        to_wav_int16(p16, convert_sample_rate(y_out, sr_in, 16000), 16000)
        outs["enhanced_16k"] = p16

    manifest = {
        "archivo_entrada": in_path,
        "etapa": "reduccion_ruido_clearvoice",
        "modelo": chosen,
        "sr_entrada_hz": int(sr_in),
        "sr_nativo_modelo_hz": int(native_sr),
        "device": device,
        "lufs_entrada": round(measure_lufs(x, sr_in), 2),
        "lufs_salida": round(measure_lufs(y_out, sr_in), 2),
        "salidas": outs,
    }
    mpath = os.path.splitext(out_path)[0] + "_denoise_manifest.json"
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    manifest["manifiesto_json"] = mpath
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entrada")
    ap.add_argument("-o", "--salida", default=None)
    ap.add_argument("--model", default="auto", choices=["auto", "FRCRN_SE_16K", "MossFormer2_SE_48K"])
    ap.add_argument("--no-16k", action="store_true", help="No generar copia adicional a 16 kHz")
    args = ap.parse_args()
    m = enhance(args.entrada, args.salida, args.model, also_16k=not args.no_16k)
    print(json.dumps(m, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
