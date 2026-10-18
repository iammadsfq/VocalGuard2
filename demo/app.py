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


def trim_silence_rms(
    y: np.ndarray,
    frame_len: int = 320,
    hop_len: int = 160,
    rel_thresh: float = 0.15,
    abs_Floor: float = 0.003,
) -> tuple[np.ndarray, float]:
    """
    Strip inter-word silence frames inside a chunk (mitigating H6 silence shortcut)
    and return (active_audio, active_frame_ratio).
    """
    if len(y) < frame_len:
        return y, 1.0
    n_f = 1 + (len(y) - frame_len) // hop_len
    frames = np.lib.stride_tricks.as_strided(
        y,
        shape=(n_f, frame_len),
        strides=(y.strides[0] * hop_len, y.strides[0]),
    )
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    peak_rms = float(np.max(rms))
    thresh = max(abs_Floor, peak_rms * rel_thresh)
    keep = rms >= thresh
    active_ratio = float(np.mean(keep))
    if int(np.sum(keep)) < 6:
        return y, active_ratio
    idx = np.where(keep)[0]
    active_y = np.concatenate([y[i * hop_len : i * hop_len + hop_len] for i in idx])
    return active_y.astype(np.float32), active_ratio


def condition_live_mic_audio(y: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Calibrate raw consumer laptop microphone audio to match ASVspoof 2019 LA studio conditions:
    1. Remove DC offset.
    2. Strip inter-word silent pauses via frame-level RMS VAD (mitigating H6 silence shortcut).
    3. Normalize active voiced segments to ASVspoof studio RMS (~0.18).
    4. Add a natural studio thermal floor (-42 dBFS) so laptop mic high-frequency roll-off (>3.8 kHz)
       does not trigger the ln(0 + 1e-12) = -27.63 cliff in upper LFCC filterbanks.
    """
    y = y - float(np.mean(y))
    y_active, active_ratio = trim_silence_rms(y, rel_thresh=0.15)
    rms = float(np.sqrt(np.mean(y_active ** 2) + 1e-12))
    y_norm = y_active * (0.18 / rms)
    y_norm = np.clip(y_norm, -0.98, 0.98)
    seed = int(abs(float(np.sum(y_norm)) * 1000.0)) % 10007
    rng = np.random.default_rng(seed)
    dither = rng.normal(0.0, 0.0075, size=len(y_norm)).astype(np.float32)
    return (y_norm + dither).astype(np.float32), active_ratio


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
        mic_conditioning: bool = False,
    ) -> dict:
        """
        Run full end-to-end pipeline on a single 1.28s audio chunk:
        1. Compute RMS energy, dBFS, and Voice Activity Detection (VAD) state
        2. Optionally apply Live Microphone Studio Conditioning (for live WebSocket mic streams)
        3. Pad/wrap chunk to produce exactly 128 LFCC frames
        4. Extract 60-dim LFCC via vocal_guard.features.extract_lfcc
        5. Run ONNX Runtime inference
        6. Return verdict (BONAFIDE / SPOOF / SILENCE), confidence, raw score, logits, latency, and RTF
        """
        if model_type not in self.sessions:
            model_type = "fp32"
        sess = self.sessions[model_type]
        in_name = sess.get_inputs()[0].name

        t_start = time.perf_counter()

        # Ensure 1D float32 and measure raw signal energy before any conditioning
        y_raw = np.asarray(audio_chunk_16k, dtype=np.float32).reshape(-1)
        rms_energy = float(np.sqrt(np.mean(y_raw ** 2) + 1e-12))
        peak_amp = float(np.max(np.abs(y_raw))) if len(y_raw) > 0 else 0.0
        db_fs = float(20.0 * np.log10(max(rms_energy, 1e-7)))

        if mic_conditioning:
            y_proc, active_ratio = condition_live_mic_audio(y_raw)
            is_speech = bool(rms_energy >= vad_rms_threshold and active_ratio >= 0.12)
        else:
            y_proc = y_raw
            _, active_ratio = trim_silence_rms(y_raw)
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
        t_feat_end = time.perf_counter()

        # ONNX Inference
        t_inf_start = time.perf_counter()
        logits = sess.run(None, {in_name: feat_tensor})[0]  # (1, 2)
        t_inf_end = time.perf_counter()

        logit_bonafide = float(logits[0, 0])
        logit_spoof = float(logits[0, 1])
        raw_score = logit_bonafide - logit_spoof

        # Calibrated sigmoid probability around decision threshold (temperature T=4.0 for smooth UI gauge)
        shifted_score = (raw_score - threshold) / 4.0
        shifted_score = float(np.clip(shifted_score, -20.0, 20.0))
        prob_bonafide = float(1.0 / (1.0 + np.exp(-shifted_score)))
        prob_spoof = float(1.0 - prob_bonafide)

        if vad_guard and not is_speech:
            verdict = "SILENCE"
            confidence_pct = 0.0
        elif raw_score >= threshold:
            verdict = "BONAFIDE"
            confidence_pct = round(prob_bonafide * 100.0, 1)
        else:
            verdict = "SPOOF"
            confidence_pct = round(prob_spoof * 100.0, 1)

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
            "confidence_pct": confidence_pct,
            "prob_bonafide_pct": round(prob_bonafide * 100.0, 1),
            "prob_spoof_pct": round(prob_spoof * 100.0, 1),
            "threshold": round(threshold, 4),
            "raw_score": round(raw_score, 4),
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


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Real-time WebSocket endpoint receiving raw Float32 PCM chunks (20,480 samples @ 16kHz)
    and returning live detection verdict + full metrics per chunk.
    Supports live side-by-side FP32 vs INT8 comparison, configurable threshold/VAD,
    and Live Microphone Studio Conditioning (default True).
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
                    res_s["start_sec"] = start_sec
                    res_s["end_sec"] = end_sec
                    res_p["comparison"] = res_s

                await websocket.send_json(res_p)
    except (WebSocketDisconnect, RuntimeError):
        pass



