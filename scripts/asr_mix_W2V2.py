# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = (PROJECT_ROOT.parent / "INTERSPEECH2023").resolve()


def parse_args():
    parser = argparse.ArgumentParser(
        description="MIX ASR přes lokální Wav2Vec2 CTC model (+ volitelný LM.arpa)."
    )
    parser.add_argument("audio_path", help="Cesta k MIX WAV souboru")
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help=f"Složka s lokálním W2V2 modelem (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--lm-path",
        default=None,
        help="Cesta k LM.arpa (volitelné). Pokud není zadáno, zkusí se <model-dir>/LM.arpa",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=25.0,
        help="Délka chunku v sekundách (default: 25.0)",
    )
    parser.add_argument(
        "--overlap-seconds",
        type=float,
        default=1.0,
        help="Překryv chunků v sekundách (default: 1.0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size pro inference (default: 4)",
    )
    parser.add_argument(
        "--start-seconds",
        type=float,
        default=None,
        help="Volitelný začátek časového výřezu v sekundách",
    )
    parser.add_argument(
        "--end-seconds",
        type=float,
        default=None,
        help="Volitelný konec časového výřezu v sekundách",
    )
    return parser.parse_args()


def resolve_time_window(start_sec=None, end_sec=None):
    if start_sec is not None and start_sec < 0:
        raise ValueError("--start-seconds musí být >= 0 nebo None")
    if end_sec is not None and end_sec < 0:
        raise ValueError("--end-seconds musí být >= 0 nebo None")
    if start_sec is not None and end_sec is not None and end_sec <= start_sec:
        raise ValueError("--end-seconds musí být větší než --start-seconds")
    return start_sec, end_sec


def to_float32(audio):
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = max(abs(info.min), info.max)
        return audio.astype(np.float32) / float(scale)
    return audio.astype(np.float32)


def load_audio_segment(file_path, start_sec=None, end_sec=None):
    sample_rate, data = wavfile.read(file_path)

    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    data = to_float32(data)

    start_sec, end_sec = resolve_time_window(start_sec, end_sec)

    start_sample = int(start_sec * sample_rate) if start_sec is not None else 0
    end_sample = int(end_sec * sample_rate) if end_sec is not None else len(data)
    end_sample = min(end_sample, len(data))

    if start_sample >= len(data):
        return np.array([], dtype=np.float32), sample_rate, start_sample / sample_rate

    segment = data[start_sample:end_sample]
    return segment, sample_rate, (start_sample / sample_rate)


def resample_if_needed(audio, src_sr, target_sr=16000):
    if src_sr == target_sr:
        return audio, src_sr

    duration = len(audio) / float(src_sr)
    target_len = int(round(duration * target_sr))
    if target_len <= 1:
        return np.array([], dtype=np.float32), target_sr

    x_old = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, duration, num=target_len, endpoint=False)
    resampled = np.interp(x_new, x_old, audio).astype(np.float32)
    return resampled, target_sr


def normalize_text(text):
    cleaned = (text or "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def build_chunks(audio, sample_rate, chunk_seconds, overlap_seconds, base_offset_sec=0.0):
    if len(audio) == 0:
        return []

    chunk_samples = max(1, int(chunk_seconds * sample_rate))
    overlap_samples = int(overlap_seconds * sample_rate)
    step_samples = max(1, chunk_samples - overlap_samples)

    chunks = []
    start = 0
    index = 1
    while start < len(audio):
        end = min(start + chunk_samples, len(audio))
        chunk_audio = audio[start:end]

        chunks.append(
            {
                "id": f"asr_{index:05d}",
                "start": round(base_offset_sec + (start / sample_rate), 3),
                "end": round(base_offset_sec + (end / sample_rate), 3),
                "audio": chunk_audio,
                "speakers": ["unknown"],
                "is_overlap": False,
            }
        )

        index += 1
        if end >= len(audio):
            break
        start += step_samples

    return chunks


def try_build_lm_decoder(processor, lm_path):
    if lm_path is None:
        return None, False, "LM path není zadán"

    lm_file = Path(lm_path)
    if not lm_file.exists():
        return None, False, f"LM soubor neexistuje: {lm_file}"

    try:
        from pyctcdecode import build_ctcdecoder
    except Exception as exc:
        return None, False, f"pyctcdecode není dostupné ({exc})"

    vocab_dict = processor.tokenizer.get_vocab()
    sorted_vocab = [token for token, _ in sorted(vocab_dict.items(), key=lambda item: item[1])]

    try:
        decoder = build_ctcdecoder(labels=sorted_vocab, kenlm_model_path=str(lm_file))
        return decoder, True, f"LM aktivní: {lm_file}"
    except Exception as exc:
        return None, False, f"LM se nepodařilo načíst ({exc})"


def decode_batch_with_lm(decoder, logits_np):
    decoded = []
    for row in logits_np:
        decoded.append(decoder.decode(row))
    return decoded


def transcribe_chunks(chunks, processor, model, device, batch_size=4, lm_decoder=None):
    if not chunks:
        return []

    results = []

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        audio_batch = [item["audio"] for item in batch]

        inputs = processor(
            audio_batch,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )

        input_values = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None

        with torch.no_grad():
            logits = model(input_values, attention_mask=attention_mask).logits

        logits_np = logits.detach().cpu().numpy()

        if lm_decoder is not None:
            texts = decode_batch_with_lm(lm_decoder, logits_np)
        else:
            pred_ids = torch.argmax(logits, dim=-1)
            texts = processor.batch_decode(pred_ids)

        for item, text in zip(batch, texts):
            item_copy = {k: v for k, v in item.items() if k != "audio"}
            item_copy["text"] = normalize_text(text)
            results.append(item_copy)

    return results


def build_output_path(audio_path, has_custom_range):
    output_scope = "range" if has_custom_range else "full"
    output_name = f"{audio_path.stem}_{output_scope}_w2v2.json"
    return PROJECT_ROOT / "results" / output_name


def run_mix_asr_w2v2():
    args = parse_args()

    run_started_utc = datetime.now(timezone.utc)
    runtime_start = time.perf_counter()

    audio_path = Path(args.audio_path).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()

    if not audio_path.exists():
        print(f"Chyba: Audio soubor neexistuje: {audio_path}")
        sys.exit(1)

    if not model_dir.exists():
        print(f"Chyba: Model složka neexistuje: {model_dir}")
        sys.exit(1)

    lm_path = args.lm_path
    if lm_path is None:
        candidate = model_dir / "LM.arpa"
        if candidate.exists():
            lm_path = str(candidate)

    start_sec, end_sec = resolve_time_window(args.start_seconds, args.end_seconds)
    has_custom_range = start_sec is not None or end_sec is not None

    print(f"Načítám audio: {audio_path}")
    audio_data, sample_rate, base_offset_sec = load_audio_segment(str(audio_path), start_sec, end_sec)
    if len(audio_data) == 0:
        print("Chyba: Zvolený rozsah neobsahuje data.")
        sys.exit(1)

    if sample_rate != 16000:
        print(f"Resampluji {sample_rate} Hz -> 16000 Hz")
        audio_data, sample_rate = resample_if_needed(audio_data, sample_rate, 16000)

    print(f"Načítám model z: {model_dir}")
    processor = Wav2Vec2Processor.from_pretrained(str(model_dir), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(model_dir), local_files_only=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    lm_decoder, lm_enabled, lm_message = try_build_lm_decoder(processor, lm_path)
    print(lm_message)

    print("Připravuji chunky...")
    chunks = build_chunks(
        audio_data,
        sample_rate,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
        base_offset_sec=base_offset_sec,
    )

    print(f"Spouštím W2V2 inference ({len(chunks)} chunků, device={device})...")
    asr_segments = transcribe_chunks(
        chunks,
        processor,
        model,
        device,
        batch_size=args.batch_size,
        lm_decoder=lm_decoder,
    )

    asr_segments = [segment for segment in asr_segments if segment.get("text")]
    full_transcription = " ".join(segment["text"] for segment in asr_segments).strip()

    runtime_seconds = round(time.perf_counter() - runtime_start, 2)
    run_finished_utc = datetime.now(timezone.utc)

    output_path = build_output_path(audio_path, has_custom_range)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    final_output = {
        "metadata": {
            "mode": "MIX",
            "start_seconds": start_sec,
            "end_seconds": end_sec,
            "has_custom_range": has_custom_range,
            "model": model_dir.name,
            "backend": "wav2vec2_ctc",
            "device": device,
            "chunk_seconds": args.chunk_seconds,
            "chunk_overlap_seconds": args.overlap_seconds,
            "batch_size": args.batch_size,
            "lm_enabled": lm_enabled,
            "lm_path": lm_path,
            "runtime_seconds": runtime_seconds,
            "run_started_utc": run_started_utc.isoformat(),
            "run_finished_utc": run_finished_utc.isoformat(),
        },
        "segments": asr_segments,
        "full_transcription": full_transcription,
    }

    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(final_output, output_file, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("MIX ASR W2V2 HOTOVO")
    print(f"Segmentů: {len(asr_segments)}")
    print(f"Výstup: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_mix_asr_w2v2()
