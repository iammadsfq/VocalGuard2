"""
Vocal-Guard Streaming Demo — FastAPI Backend
Implements PRD (17 Oct 2026):
- Chunk-by-chunk (1.28s = 20,480 samples @ 16kHz -> 128 LFCC frames) processing
- Uses vocal_guard/features.py as Single Source of Truth for 60-dim LFCC
- Supports ONNX Runtime FP32 (default, fastest on non-VNNI x86 CPU) and ONNX Dynamic INT8
- WebSocket endpoint (/ws/stream) for live microphone chunks or streamed file chunks
- REST endpoint (/api/analyze-file) for deterministic chunk-by-chunk file processing
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import onnxruntime as ort
import scipy.signal
import soundfile as sf

from vocal_guard.dataset import pad_or_truncate
from vocal_guard.features import extract_lfcc

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
DEMO_DIR = ROOT_DIR / "demo"
SAMPLES_DIR = DEMO_DIR / "samples"

TARGET_SR = 16000
CHUNK_DURATION_SEC = 1.28
CHUNK_SAMPLES = 20480  # 1.28s * 16,000 Hz
# For 20ms window (320) and 10ms hop (160), 128 frames requires 20,640 samples.
# We pad/wrap 20,480 samples to 20,640 so extract_lfcc yields exactly (128, 60).
LFCC_REQUIRED_SAMPLES_128F = 20640
TARGET_FRAMES = 128


def compact_voiced_frames(
    y: np.ndarray,
    frame_len: int = 320,
    hop_len: int = 160,
    rel_thresh: float = 0.22,
    abs_floor: float = 0.0075,
) -> tuple[np.ndarray, float]:
    """
    Strip inter-word silence and low-SNR room noise frames (< -42 dBFS or < 22% of peak syllable RMS)
    inside a 1.28s chunk WITHOUT injecting artificial Gaussian white noise dither.
    Preserves true high-frequency spectral holes of TTS/Replay while preventing pause-induced H6 false spoofs.
    """
    y = y - float(np.mean(y))
    if len(y) < frame_len:
        return y.astype(np.float32), 1.0
    n_f = 1 + (len(y) - frame_len) // hop_len
    frames = np.lib.stride_tricks.as_strided(
        y,
        shape=(n_f, frame_len),
        strides=(y.strides[0] * hop_len, y.strides[0]),
    )
    f_rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    peak_rms = float(np.max(f_rms))
    thresh = max(abs_floor, peak_rms * rel_thresh)
    keep = f_rms >= thresh
    active_ratio = float(np.mean(keep))
    if int(np.sum(keep)) < 8:
        keep = f_rms >= max(0.0035, peak_rms * 0.15)
        active_ratio = float(np.mean(keep))
    if int(np.sum(keep)) < 6:
        return y.astype(np.float32), active_ratio
    idx = np.where(keep)[0]
    voiced_y = np.concatenate([y[i * hop_len : i * hop_len + hop_len] for i in idx])
    return voiced_y.astype(np.float32), active_ratio


def compute_pa_tts_penalty(y_voiced: np.ndarray, sr: int = TARGET_SR) -> tuple[float, dict]:
    """
    Hybrid Physical Access (PA Loudspeaker Replay) + Modern Neural TTS (Gemini / Google Maps Voiceover)
    acoustic detector to complement the ASVspoof 2019 LA LCNN model on live microphone streams:
    1. Transducer Bass Roll-off & Mid-Horn Resonance (f0_to_horn, low_to_horn, horn_frac):
       Phone/laptop loudspeakers physically attenuate fundamental F0 (75-340 Hz) and concentrate
       acoustic energy in the 800-3,800 Hz horn resonance band.
    2. Neural TTS Pitch Over-Smoothing & Broadcast AGC Compression (jitter, std_nacf, env_cv):
       Neural TTS voiceovers (Gemini, Google Maps) exhibit ultra-smooth pitch trajectories and
       narrow dynamic range compared to spontaneous human speech.
    """
    if len(y_voiced) < 1600:
        return 0.0, {"f0_to_horn": 0.0, "low_to_horn": 0.0, "horn_frac": 0.0, "jitter": 0.0, "env_cv": 0.0}

    yv = y_voiced - float(np.mean(y_voiced))
    yv_norm = yv / (float(np.sqrt(np.mean(yv ** 2))) + 1e-9)

    freqs, psd = scipy.signal.welch(yv_norm, fs=sr, nperseg=min(512, len(yv_norm)))
    e_tot = float(np.sum(psd[(freqs >= 60) & (freqs <= 7800)]) + 1e-12)
    e_f0 = float(np.sum(psd[(freqs >= 75) & (freqs <= 340)]) + 1e-12)
    e_lowmid = float(np.sum(psd[(freqs > 340) & (freqs <= 780)]) + 1e-12)
    e_horn = float(np.sum(psd[(freqs > 800) & (freqs <= 3800)]) + 1e-12)

    f0_to_horn = e_f0 / e_horn
    low_to_horn = (e_f0 + 0.65 * e_lowmid) / e_horn
    horn_frac = e_horn / e_tot

    n_f = max(1, len(yv_norm) // 320)
    frames = yv_norm[: n_f * 320].reshape(n_f, 320)
    f_rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    env_cv = float(np.std(f_rms) / (np.mean(f_rms) + 1e-9))

    nacf_peaks = []
    lags = []
    for fr in frames:
        fr0 = fr - np.mean(fr)
        nrm = float(np.sum(fr0 ** 2)) + 1e-12
        acf = np.correlate(fr0, fr0, mode="full")[319:] / nrm
        sub = acf[32:200]
        pk = int(np.argmax(sub))
        val = float(sub[pk])
        nacf_peaks.append(val)
        if val > 0.45:
            lags.append(pk + 32)
    std_nacf = float(np.std(nacf_peaks)) if len(nacf_peaks) > 1 else 0.05
    if len(lags) >= 4:
        diffs = np.abs(np.diff(lags)) / (np.array(lags[:-1], dtype=np.float32) + 1e-6)
        valid = diffs[diffs < 0.22]
        jitter = float(np.mean(valid)) if len(valid) > 0 else 0.015
    else:
        jitter = 0.015

    penalty = 0.0
    # 1. Loudspeaker / Phone Speaker Bass Roll-off & Horn Concentration (PA Replay)
    if f0_to_horn < 4.20:
        penalty -= (4.20 - f0_to_horn) * 5.5
    if low_to_horn < 5.00:
        penalty -= (5.00 - low_to_horn) * 4.5
    if horn_frac > 0.12:
        penalty -= (horn_frac - 0.12) * 55.0

    # 2. Neural TTS Pitch Regularity & Broadcast AGC Envelope Compression (Gemini / GMaps Voiceover)
    if jitter < 0.0080 and std_nacf < 0.045:
        penalty -= (0.0085 - jitter) * 850.0
    if env_cv < 0.27 and horn_frac > 0.11:
        penalty -= (0.28 - env_cv) * 65.0

    cues = {
        "f0_to_horn": round(f0_to_horn, 2),
        "low_to_horn": round(low_to_horn, 2),
        "horn_frac": round(horn_frac, 3),
        "jitter": round(jitter, 4),
        "env_cv": round(env_cv, 2),
    }
    return float(penalty), cues


class ONNXChunkInferenceEngine:
    """Manages single-threaded ONNX Runtime sessions for FP32 and INT8 models."""

    def __init__(self):
        self.sessions: dict[str, ort.InferenceSession] = {}
        self.model_paths = {
            "fp32": MODELS_DIR / "lcnn_m1_128frames.onnx",
            "int8": MODELS_DIR / "lcnn_m1_128frames_int8.onnx",
        }
        self._init_sessions()

    def _init_sessions(self):
        for key, path in self.model_paths.items():
            if not path.exists():
                raise FileNotFoundError(f"Required ONNX model not found: {path}")
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
            # Warmup session
            dummy = np.zeros((1, 1, 60, TARGET_FRAMES), dtype=np.float32)
            in_name = sess.get_inputs()[0].name
            for _ in range(3):
                sess.run(None, {in_name: dummy})
            self.sessions[key] = sess

    def process_chunk(
        self,
        audio_chunk_16k: np.ndarray,
        model_type: Literal["fp32", "int8"] = "fp32",
        chunk_index: int = 0,
        threshold: float = 0.0,
        vad_guard: bool = True,
        vad_rms_threshold: float = 0.007,
        mic_conditioning: bool = True,
    ) -> dict:
        """
        Run full end-to-end pipeline on a single 1.28s audio chunk:
        1. Compute RMS energy, dBFS, and Voice Activity Detection (VAD) state
        2. Compact voiced frames (stripping inter-word silence without artificial white noise dither)
        3. Extract 60-dim LFCC via vocal_guard.features.extract_lfcc & run ONNX LCNN inference
        4. Compute Hybrid PA Replay + Neural TTS penalty (for phone/speaker replay of Gemini/GMaps TTS)
        5. Return verdict (BONAFIDE / SPOOF / SILENCE), confidence, scores, cues, latency, and RTF
        """
        if model_type not in self.sessions:
            model_type = "fp32"
        sess = self.sessions[model_type]
        in_name = sess.get_inputs()[0].name

        t_start = time.perf_counter()

        # Ensure 1D float32 and measure raw signal energy before compaction
        y_raw = np.asarray(audio_chunk_16k, dtype=np.float32).reshape(-1)
        rms_energy = float(np.sqrt(np.mean(y_raw ** 2) + 1e-12))
        peak_amp = float(np.max(np.abs(y_raw))) if len(y_raw) > 0 else 0.0
        db_fs = float(20.0 * np.log10(max(rms_energy, 1e-7)))

        y_voiced, active_ratio = compact_voiced_frames(y_raw)
        if mic_conditioning:
            y_proc = y_voiced
            is_speech = bool(rms_energy >= vad_rms_threshold and active_ratio >= 0.14)
        else:
            y_proc = y_raw
            is_speech = bool(rms_energy >= vad_rms_threshold)

        y_padded = pad_or_truncate(y_proc, target_length=LFCC_REQUIRED_SAMPLES_128F)

        t_feat_start = time.perf_counter()
        lfcc = extract_lfcc(y_padded, sr=TARGET_SR, with_deltas=True)  # (128, 60)
        if lfcc.shape[0] != TARGET_FRAMES:
            if lfcc.shape[0] > TARGET_FRAMES:
                lfcc = lfcc[:TARGET_FRAMES, :]
            else:
                lfcc = np.pad(lfcc, ((0, TARGET_FRAMES - lfcc.shape[0]), (0, 0)))
        feat_tensor = np.expand_dims(np.expand_dims(lfcc.T.astype(np.float32), axis=0), axis=0)  # (1, 1, 60, 128)
        pa_penalty, pa_cues = compute_pa_tts_penalty(y_voiced, sr=TARGET_SR) if mic_conditioning else (0.0, {})
        t_feat_end = time.perf_counter()

        # ONNX Inference
        t_inf_start = time.perf_counter()
        logits = sess.run(None, {in_name: feat_tensor})[0]  # (1, 2)
        t_inf_end = time.perf_counter()

        logit_bonafide = float(logits[0, 0])
        logit_spoof = float(logits[0, 1])
        la_score = logit_bonafide - logit_spoof
        raw_score = la_score + pa_penalty

        # Calibrated sigmoid probability around decision threshold (temperature T=4.0 for smooth UI gauge)
        shifted_score = (raw_score - threshold) / 4.0
        shifted_score = float(np.clip(shifted_score, -20.0, 20.0))
        prob_bonafide = float(1.0 / (1.0 + np.exp(-shifted_score)))
        prob_spoof = float(1.0 - prob_bonafide)

        if vad_guard and not is_speech:
            verdict = "SILENCE"
            confidence_pct = 0.0
            spoof_reason = "SILENCE"
        elif raw_score >= threshold:
            verdict = "BONAFIDE"
            confidence_pct = round(prob_bonafide * 100.0, 1)
            spoof_reason = "DIRECT_VOICE"
        else:
            verdict = "SPOOF"
            confidence_pct = round(prob_spoof * 100.0, 1)
            if la_score < threshold and pa_penalty < -2.5:
                spoof_reason = "HYBRID_LA_PA"
            elif pa_penalty < -2.5:
                spoof_reason = "PA_REPLAY_TTS"
            else:
                spoof_reason = "LA_VOCODER"

        feat_ms = (t_feat_end - t_feat_start) * 1000.0
        infer_ms = (t_inf_end - t_inf_start) * 1000.0
        total_ms = (t_inf_end - t_start) * 1000.0

        rtf_infer = (infer_ms / 1000.0) / CHUNK_DURATION_SEC
        rtf_total = (total_ms / 1000.0) / CHUNK_DURATION_SEC

        return {
            "chunk_index": chunk_index,
            "duration_sec": CHUNK_DURATION_SEC,
            "model_type": model_type.upper(),
            "verdict": verdict,
            "spoof_reason": spoof_reason,
            "confidence_pct": confidence_pct,
            "prob_bonafide_pct": round(prob_bonafide * 100.0, 1),
            "prob_spoof_pct": round(prob_spoof * 100.0, 1),
            "threshold": round(threshold, 4),
            "raw_score": round(raw_score, 4),
            "la_score": round(la_score, 4),
            "pa_penalty": round(pa_penalty, 4),
            "pa_cues": pa_cues,
            "logit_bonafide": round(logit_bonafide, 4),
            "logit_spoof": round(logit_spoof, 4),
            "feature_ms": round(feat_ms, 2),
            "inference_ms": round(infer_ms, 2),
            "total_ms": round(total_ms, 2),
            "rtf_inference": round(rtf_infer, 5),
            "rtf_total": round(rtf_total, 5),
            "rms_energy": round(rms_energy, 5),
            "peak_amp": round(peak_amp, 4),
            "db_fs": round(db_fs, 1),
            "active_speech_ratio": round(active_ratio, 2),
            "is_speech": is_speech,
            "mic_conditioning": mic_conditioning,
        }



app = FastAPI(title="Vocal-Guard Streaming Demo", version="0.2.0")
engine = ONNXChunkInferenceEngine()


def load_and_resample_to_16k_mono(raw_bytes: bytes) -> tuple[np.ndarray, float]:
    """Read audio bytes (WAV/FLAC/OGG), convert to mono, and resample to 16kHz."""
    data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    orig_duration = len(data) / float(sr) if sr > 0 else 0.0
    if sr != TARGET_SR:
        target_len = int(round(len(data) * float(TARGET_SR) / float(sr)))
        data = scipy.signal.resample(data, target_len).astype(np.float32)
    return data, orig_duration


def summarize_chunks(chunks: list[dict], model_type: str, threshold: float = 0.0) -> dict:
    if not chunks:
        return {}
    inf_ms = np.array([c["inference_ms"] for c in chunks])
    tot_ms = np.array([c["total_ms"] for c in chunks])
    feat_ms = np.array([c["feature_ms"] for c in chunks])
    scores = np.array([c["raw_score"] for c in chunks])
    rtfs = np.array([c["rtf_total"] for c in chunks])
    rtfs_inf = np.array([c["rtf_inference"] for c in chunks])

    active_chunks = [c for c in chunks if c.get("verdict") != "SILENCE"]
    bonafide_count = sum(1 for c in chunks if c.get("verdict") == "BONAFIDE")
    spoof_count = sum(1 for c in chunks if c.get("verdict") == "SPOOF")
    silence_count = sum(1 for c in chunks if c.get("verdict") == "SILENCE")

    if active_chunks:
        active_scores = np.array([c["raw_score"] for c in active_chunks])
        mean_active_score = float(np.mean(active_scores))
        session_verdict = "BONAFIDE" if mean_active_score >= threshold else "SPOOF"
    else:
        mean_active_score = float(np.mean(scores))
        session_verdict = "SILENCE"

    model_size_mb = 2.08 if model_type.lower() == "fp32" else 0.56
    return {
        "num_chunks": len(chunks),
        "bonafide_chunks": bonafide_count,
        "spoof_chunks": spoof_count,
        "silence_chunks": silence_count,
        "session_verdict": session_verdict,
        "model_type": model_type.upper(),
        "model_size_mb": model_size_mb,
        "mean_feature_ms": round(float(np.mean(feat_ms)), 2),
        "mean_inference_ms": round(float(np.mean(inf_ms)), 2),
        "p95_inference_ms": round(float(np.percentile(inf_ms, 95)), 2),
        "mean_total_ms": round(float(np.mean(tot_ms)), 2),
        "p95_total_ms": round(float(np.percentile(tot_ms, 95)), 2),
        "mean_rtf_inference": round(float(np.mean(rtfs_inf)), 5),
        "mean_rtf": round(float(np.mean(rtfs)), 5),
        "mean_raw_score": round(float(np.mean(scores)), 4),
        "mean_active_score": round(mean_active_score, 4),
        "min_raw_score": round(float(np.min(scores)), 4),
        "max_raw_score": round(float(np.max(scores)), 4),
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    index_file = DEMO_DIR / "static" / "index.html"
    return index_file.read_text(encoding="utf-8")


@app.get("/api/samples")
async def list_samples():
    """Return available built-in reference audio samples."""
    files = []
    if SAMPLES_DIR.exists():
        for p in sorted(SAMPLES_DIR.glob("*.wav")):
            files.append({
                "name": p.name,
                "url": f"/api/samples/{p.name}",
            })
    return {"samples": files}


@app.get("/api/samples/{filename}")
async def get_sample_file(filename: str):
    safe_name = Path(filename).name
    target = SAMPLES_DIR / safe_name
    if not target.exists():
        raise HTTPException(status_code=404, detail="Sample file not found")
    return FileResponse(str(target), media_type="audio/wav", filename=safe_name)


@app.post("/api/analyze-file")
async def analyze_file(
    file: UploadFile = File(...),
    model_type: str = Form("fp32"),
    compare_both: bool = Form(False),
    threshold: float = Form(0.0),
    vad_guard: bool = Form(True),
):
    """
    Slice uploaded audio file into 1.28s (20,480 sample) chunks and run chunk-by-chunk
    LFCC + ONNX inference. Optionally runs side-by-side FP32 vs INT8 comparison.
    """
    raw_bytes = await file.read()
    try:
        audio_16k, duration_sec = load_and_resample_to_16k_mono(raw_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Gagal membaca file audio: {exc}") from exc

    if len(audio_16k) == 0:
        raise HTTPException(status_code=400, detail="File audio kosong.")

    num_chunks = max(1, int(np.ceil(len(audio_16k) / CHUNK_SAMPLES)))
    m_type: Literal["fp32", "int8"] = "int8" if model_type.lower() == "int8" else "fp32"

    chunks_primary = []
    chunks_secondary = []

    for i in range(num_chunks):
        start = i * CHUNK_SAMPLES
        end = start + CHUNK_SAMPLES
        segment = audio_16k[start:end]
        if len(segment) < CHUNK_SAMPLES:
            segment = pad_or_truncate(segment, target_length=CHUNK_SAMPLES)

        res_p = engine.process_chunk(
            segment,
            model_type=m_type,
            chunk_index=i + 1,
            threshold=threshold,
            vad_guard=vad_guard,
        )
        res_p["start_sec"] = round(i * CHUNK_DURATION_SEC, 2)
        res_p["end_sec"] = round((i + 1) * CHUNK_DURATION_SEC, 2)
        chunks_primary.append(res_p)

        if compare_both:
            other_type: Literal["fp32", "int8"] = "int8" if m_type == "fp32" else "fp32"
            res_s = engine.process_chunk(
                segment,
                model_type=other_type,
                chunk_index=i + 1,
                threshold=threshold,
                vad_guard=vad_guard,
            )
            res_s["start_sec"] = res_p["start_sec"]
            res_s["end_sec"] = res_p["end_sec"]
            chunks_secondary.append(res_s)

    response = {
        "filename": file.filename,
        "duration_sec": round(duration_sec, 2),
        "sample_rate": TARGET_SR,
        "chunk_duration_sec": CHUNK_DURATION_SEC,
        "primary": {
            "summary": summarize_chunks(chunks_primary, m_type, threshold=threshold),
            "chunks": chunks_primary,
        },
    }
    if compare_both and chunks_secondary:
        other_label = "int8" if m_type == "fp32" else "fp32"
        score_diffs = [
            abs(p["raw_score"] - s["raw_score"])
            for p, s in zip(chunks_primary, chunks_secondary)
        ]
        response["comparison"] = {
            "summary": summarize_chunks(chunks_secondary, other_label, threshold=threshold),
            "chunks": chunks_secondary,
            "mean_abs_score_diff": round(float(np.mean(score_diffs)), 4),
            "max_abs_score_diff": round(float(np.max(score_diffs)), 4),
        }

    return response


def _apply_ema_to_chunk(res: dict, prev_ema: float | None, threshold: float, alpha: float = 0.65) -> float | None:
    """Smooth active speech chunk scores with a 2-chunk EMA to prevent single-hop boundary jitter."""
    if res.get("verdict") == "SILENCE":
        return prev_ema
    cur = float(res["raw_score"])
    ema = cur if prev_ema is None else (alpha * cur + (1.0 - alpha) * prev_ema)
    res["raw_score_unsmoothed"] = round(cur, 4)
    res["raw_score"] = round(ema, 4)
    shifted = float(np.clip((ema - threshold) / 4.0, -20.0, 20.0))
    prob_bona = float(1.0 / (1.0 + np.exp(-shifted)))
    prob_spoof = 1.0 - prob_bona
    res["prob_bonafide_pct"] = round(prob_bona * 100.0, 1)
    res["prob_spoof_pct"] = round(prob_spoof * 100.0, 1)
    if ema >= threshold:
        res["verdict"] = "BONAFIDE"
        res["confidence_pct"] = round(prob_bona * 100.0, 1)
        res["spoof_reason"] = "DIRECT_VOICE"
    else:
        res["verdict"] = "SPOOF"
        res["confidence_pct"] = round(prob_spoof * 100.0, 1)
        if res.get("spoof_reason") == "DIRECT_VOICE":
            res["spoof_reason"] = "PA_REPLAY_TTS" if res.get("pa_penalty", 0.0) < -2.0 else "LA_VOCODER"
    return ema


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Real-time WebSocket endpoint receiving raw Float32 PCM chunks (20,480 samples @ 16kHz)
    and returning live detection verdict + full metrics per chunk.
    Supports live side-by-side FP32 vs INT8 comparison, configurable threshold/VAD,
    and Hybrid LA + PA/TTS Live Microphone Conditioning (default True).
    """
    await websocket.accept()
    q = websocket.query_params
    model_type_param = q.get("model_type", "fp32").lower()
    m_type: Literal["fp32", "int8"] = "int8" if model_type_param == "int8" else "fp32"
    compare_both: bool = q.get("compare_both", "true").lower() == "true"
    threshold: float = float(q.get("threshold", "0.0"))
    vad_guard: bool = q.get("vad_guard", "true").lower() == "true"
    mic_conditioning: bool = q.get("mic_conditioning", "true").lower() == "true"
    hop_sec: float = float(q.get("hop_sec", "0.50"))
    chunk_idx = 0
    ema_p: float | None = None
    ema_s: float | None = None

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if "text" in message and message["text"]:
                import json
                try:
                    cmd = json.loads(message["text"])
                    if "model_type" in cmd:
                        m_type = "int8" if str(cmd["model_type"]).lower() == "int8" else "fp32"
                    if "compare_both" in cmd:
                        compare_both = bool(cmd["compare_both"])
                    if "threshold" in cmd:
                        threshold = float(cmd["threshold"])
                    if "vad_guard" in cmd:
                        vad_guard = bool(cmd["vad_guard"])
                    if "mic_conditioning" in cmd:
                        mic_conditioning = bool(cmd["mic_conditioning"])
                    if "hop_sec" in cmd:
                        hop_sec = float(cmd["hop_sec"])
                    if cmd.get("action") == "reset":
                        chunk_idx = 0
                        ema_p = None
                        ema_s = None
                    await websocket.send_json({
                        "type": "config_ack",
                        "model_type": m_type.upper(),
                        "compare_both": compare_both,
                        "threshold": threshold,
                        "vad_guard": vad_guard,
                        "mic_conditioning": mic_conditioning,
                    })
                except Exception:
                    pass
            elif "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                audio_f32 = np.frombuffer(raw_bytes, dtype=np.float32)
                if len(audio_f32) == 0:
                    continue
                chunk_idx += 1
                end_sec = round(chunk_idx * hop_sec, 2)
                start_sec = round(max(0.0, end_sec - CHUNK_DURATION_SEC), 2)

                res_p = engine.process_chunk(
                    audio_f32,
                    model_type=m_type,
                    chunk_index=chunk_idx,
                    threshold=threshold,
                    vad_guard=vad_guard,
                    mic_conditioning=mic_conditioning,
                )
                if mic_conditioning:
                    ema_p = _apply_ema_to_chunk(res_p, ema_p, threshold=threshold)
                res_p["start_sec"] = start_sec
                res_p["end_sec"] = end_sec
                res_p["type"] = "chunk_result"

                if compare_both:
                    other_type: Literal["fp32", "int8"] = "int8" if m_type == "fp32" else "fp32"
                    res_s = engine.process_chunk(
                        audio_f32,
                        model_type=other_type,
                        chunk_index=chunk_idx,
                        threshold=threshold,
                        vad_guard=vad_guard,
                        mic_conditioning=mic_conditioning,
                    )
                    if mic_conditioning:
                        ema_s = _apply_ema_to_chunk(res_s, ema_s, threshold=threshold)
                    res_s["start_sec"] = start_sec
                    res_s["end_sec"] = end_sec
                    res_p["comparison"] = res_s

                await websocket.send_json(res_p)
    except (WebSocketDisconnect, RuntimeError):
        pass



