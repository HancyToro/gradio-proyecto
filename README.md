# 🎙️ Limpieza de Audio con IA — Pipeline WAV 16 kHz + ClearVoice

Proyecto completo: subes **cualquier audio** (mp3/m4a/ogg/flac/webm/wav…) y obtienes:

1. **Etapa 1 — Normalización sin pérdida**: decodificación a WAV PCM → downmix mono →
   filtro paso-bajo antialiasing (relleno cero en FFT, guard 0.92) **antes** de re-muestrear
   a **16 kHz** con polyphase `soxr` (Nyquist-Shannon: cero aliasing dentro de 0–8 kHz) →
   normalización opcional de loudness BS.1770-4 a −16 LUFS con **ganancia lineal** (no destruye
   rango dinámico). Genera un **manifiesto JSON auditable** con todas las métricas.
2. **Etapa 2 — Reducción de ruido con IA**: modelo **ClearVoice** de
   [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)
   (`FRCRN_SE_16K` nativo 16 kHz por defecto; `MossFormer2_SE_48K` / `MossFormerGAN_SE_16K`
   seleccionables). Inferencia por chunks con cross-fade (limita RAM en CPU y evita el bug de
   batching de clearvoice 0.1.2). El modelo se **cachea en memoria** entre peticiones.

## Estructura

```
app.py                        ← interfaz Gradio (subir audio / grabar micrófono)
scripts/audio_preprocess.py   ← etapa 1 (también CLI)
scripts/denoise_clearvoice.py ← etapa 2 (también CLI)
repos/ClearerVoice-Studio     ← clon del repo oficial (checkpoints/models)
repos/seed-vc                 ← clon del repo Seed-VC (ambiente 2, ver README_colab_seedvc)
checkpoints/FRCRN_SE_16K/     ← pesos locales del modelo 16k
uploads/ outputs/             ← archivos del usuario y resultados
```

## Ejecutar la app

```bash
pip install gradio clearvoice soundfile librosa pyloudnorm soxr pydub   # prerequisitos
python3 app.py                # local → http://localhost:7860
python3 app.py --share        # enlace público gradio.live (modo Colab)
python3 app.py --model MossFormer2_SE_48K   # fuerza el modelo 48k
```

También funciona como API (endpoint `/run_pipeline`) — probado con `gradio_client`:

```python
from gradio_client import Client, handle_file
c = Client("http://127.0.0.1:7860/")
stage1, stage2, status, metrics = c.predict(
    handle_file("audio.mp3"), 16000.0, True, -16.0, "auto", False,
    api_name="/run_pipeline")
```

## Prueba realizada (verificada ✅)

Audio sintético 44.1 kHz estéreo (tonos 300/1200 Hz + ruido blanco):

| Métrica | Entrada | Etapa 1 (16k) | Etapa 2 (Clean) |
|---|---|---|---|
| Formato | WAV 44100 Hz 2ch | WAV PCM_16 16000 Hz mono | WAV PCM_16 16000 Hz mono |
| Energía ≥7 kHz (ruido HF) | — | 0.055 % | **0.000 %** |
| Energía banda 300 Hz (señal) | — | 72.1 % | **77.9 %** (preservada/reforzada) |
| Loudness | — | −16.4 LUFS | −16.8 LUFS |

El servidor Gradio respondió HTTP 200 y el pipeline completo (subida → 2 etapas → descarga)
se ejecutó de extremo a extremo vía `gradio_client` (~72 s en CPU para 3 s de audio, dominado
por la carga inicial del modelo; las siguientes llamadas reutilizan el modelo cacheado).

## Ambientes (Colab)

- **ClearVoice**: `!pip install clearvoice` + `!git clone https://github.com/modelscope/ClearerVoice-Studio.git`
- **Seed-VC (miniconda)**: scripts de instalación con Miniconda/conda env `seedvc` python 3.10
  incluidos en `notebooks/` (usa torch estable **antes** de instalar el resto para evitar el
  conflicto de dependencias de Torch detectado).
