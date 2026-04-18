# -*- coding: utf-8 -*-
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import auditok
import numpy as np
import torch
from scipy.io import wavfile
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

START_SECONDS = None
END_SECONDS = None
CHUNK_SECONDS = 60
CHUNK_BACKSHIFT_SECONDS = 1.0
BOUNDARY_MAX_OVERLAP_WORDS = 8
DEDUP_TIME_TOLERANCE = 1.0
VAD_MIN_DUR = 0.25
VAD_MAX_DUR = 20
VAD_MAX_SILENCE = 1
VAD_ENERGY_THRESHOLD = 45

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = (PROJECT_ROOT.parent / "INTERSPEECH2023").resolve()

HALLUCINATION_PATTERNS = [
    r"\btitulky vytvořil\b",
    r"\bděkuji za sledování\b",
    r"\bthanks for watching\b",
    r"\bwatch next\b",
    r"\bjohnyx\b",
]


def clean_segment_text(text):
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    for pattern in HALLUCINATION_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,-")
    return cleaned


def normalize_for_dedup(text):
    normalized = (text or "").lower()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return " ".join(normalized.split()).strip()


def normalize_word_for_overlap(word):
    return re.sub(r"[^\w]", "", (word or "").lower())


def trim_boundary_overlap(previous_text, current_text, max_overlap_words):
    if not previous_text or not current_text or max_overlap_words <= 0:
        return current_text, 0

    previous_words = previous_text.split()
    current_words = current_text.split()
    if not previous_words or not current_words:
        return current_text, 0

    previous_norm = [normalize_word_for_overlap(word) for word in previous_words]
    current_norm = [normalize_word_for_overlap(word) for word in current_words]

    max_k = min(max_overlap_words, len(previous_norm), len(current_norm))
    for token_count in range(max_k, 0, -1):
        if previous_norm[-token_count:] == current_norm[:token_count]:
            trimmed = " ".join(current_words[token_count:]).strip()
            return trimmed, token_count

    return current_text, 0


def resolve_time_window(start_sec=None, end_sec=None):
    if start_sec is not None and start_sec < 0:
        raise ValueError("START_SECONDS musí být >= 0 nebo None")
    if end_sec is not None and end_sec < 0:
        raise ValueError("END_SECONDS musí být >= 0 nebo None")
    if start_sec is not None and end_sec is not None and end_sec <= start_sec:
        raise ValueError("END_SECONDS musí být větší než START_SECONDS")

    return start_sec, end_sec


def normalize_audio_data(data):
    if data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    if data.dtype == np.int32:
        return data.astype(np.float32) / 2147483648.0
    if data.dtype == np.uint8:
        return (data.astype(np.float32) - 128.0) / 128.0
    return data.astype(np.float32)


def load_stereo_audio_scipy(file_path, start_sec=None, end_sec=None):
    sample_rate, data = wavfile.read(file_path)

    if len(data.shape) == 1 or data.shape[1] < 2:
        raise ValueError("Vstupní soubor musí být stereo (2 kanály).")

    start_sec, end_sec = resolve_time_window(start_sec, end_sec)

    start_sample = int(start_sec * sample_rate) if start_sec is not None else 0
    end_sample = int(end_sec * sample_rate) if end_sec is not None else len(data)
    end_sample = min(end_sample, len(data))

    if start_sample >= len(data):
        left = np.array([], dtype=np.float32)
        right = np.array([], dtype=np.float32)
    else:
        data = data[start_sample:end_sample]
        left = normalize_audio_data(data[:, 0])
        right = normalize_audio_data(data[:, 1])

    return left, right, sample_rate, (start_sample / sample_rate)


def detect_speech_windows_with_auditok(audio_data, sample_rate):
    if len(audio_data) == 0:
        return []

    def read_region_times(region):
        meta = getattr(region, "meta", None)

        start = None
        end = None
        duration = None

        if meta is not None:
            if isinstance(meta, dict):
                start = meta.get("start")
                end = meta.get("end")
                duration = meta.get("duration")
            else:
                start = getattr(meta, "start", None)
                end = getattr(meta, "end", None)
                duration = getattr(meta, "duration", None)

        if start is None:
            start = getattr(region, "start", None)
        if end is None:
            end = getattr(region, "end", None)
        if duration is None:
            duration = getattr(region, "duration", None)

        return start, end, duration

    pcm16 = np.round(np.clip(audio_data, -1.0, 1.0) * 32767.0).astype(np.int16)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        wavfile.write(temp_path, sample_rate, pcm16)
        regions = auditok.split(
            temp_path,
            min_dur=VAD_MIN_DUR,
            max_dur=VAD_MAX_DUR,
            max_silence=VAD_MAX_SILENCE,
            energy_threshold=VAD_ENERGY_THRESHOLD,
        )

        windows = []
        cursor = 0.0
        audio_duration = len(audio_data) / float(sample_rate)
        for region in regions:
            start, end, duration = read_region_times(region)

            if start is None and end is None:
                if duration is None:
                    continue
                start = cursor
                end = cursor + float(duration)
            elif start is None and end is not None:
                if duration is None:
                    start = cursor
                else:
                    start = float(end) - float(duration)
            elif end is None and start is not None:
                if duration is None:
                    continue
                end = float(start) + float(duration)

            start = float(start)
            end = float(end)
            start = max(0.0, start)
            end = min(audio_duration, end)
            if end > start:
                windows.append((start, end))
                cursor = max(cursor, end)

        return windows
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def build_vad_chunks(speech_windows, audio_duration, chunk_seconds):
    if not speech_windows:
        speech_windows = [(0.0, audio_duration)]

    chunks = []
    current_start = None
    current_end = None

    for raw_start, raw_end in sorted(speech_windows, key=lambda item: item[0]):
        start = max(0.0, float(raw_start))
        end = min(audio_duration, float(raw_end))
        if end <= start:
            continue

        if current_start is None:
            current_start = start
            current_end = end
        else:
            proposed_end = end
            if proposed_end - current_start <= chunk_seconds:
                current_end = max(current_end, end)
            else:
                chunks.append((current_start, current_end))
                current_start = start
                current_end = end

        while current_end - current_start > chunk_seconds:
            split_end = current_start + chunk_seconds
            chunks.append((current_start, split_end))
            current_start = split_end

    if current_start is not None and current_end is not None and current_end > current_start:
        chunks.append((current_start, current_end))

    return chunks


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


def load_unigrams_utf8(lm_path):
    text = Path(lm_path).read_text(encoding="utf-8")
    in_1grams = False
    unigrams = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("\\1-grams:"):
            in_1grams = True
            continue

        if line.startswith("\\2-grams:") or line.startswith("\\end\\"):
            break

        if not in_1grams:
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        token = parts[1].strip()
        if token and token not in {"<s>", "</s>", "<unk>"}:
            unigrams.append(token)

    return sorted(set(unigrams))


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
    labels = [token for token, _ in sorted(vocab_dict.items(), key=lambda item: item[1])]

    try:
        unigrams = load_unigrams_utf8(lm_file)
        decoder = build_ctcdecoder(
            labels=labels,
            kenlm_model_path=str(lm_file),
            unigrams=unigrams,
        )
        return decoder, True, f"LM aktivní: {lm_file} (UTF-8 unigrams={len(unigrams)})"
    except Exception as exc:
        return None, False, f"LM se nepodařilo načíst ({exc})"


def decode_chunk_text(chunk_audio, sample_rate, processor, model, device, lm_decoder=None):
    if len(chunk_audio) == 0:
        return ""

    if sample_rate != 16000:
        chunk_audio, sample_rate = resample_if_needed(chunk_audio, sample_rate, target_sr=16000)
        if len(chunk_audio) == 0:
            return ""

    inputs = processor(
        chunk_audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )

    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    if lm_decoder is not None:
        text = lm_decoder.decode(logits[0].detach().cpu().numpy())
    else:
        pred_ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(pred_ids)[0]

    return clean_segment_text(text)


def transcribe_channel(processor, model, device, audio_data, sample_rate, speaker_tag, base_offset_sec=0.0, lm_decoder=None):
    segments = []
    audio_duration = len(audio_data) / float(sample_rate)

    speech_windows = detect_speech_windows_with_auditok(audio_data, sample_rate)
    vad_chunks = build_vad_chunks(speech_windows, audio_duration, CHUNK_SECONDS)
    total_chunks = len(vad_chunks)

    for chunk_index, (chunk_start_sec, chunk_end_sec) in enumerate(vad_chunks, start=1):
        decode_start_sec = chunk_start_sec
        if chunk_index > 1 and CHUNK_BACKSHIFT_SECONDS > 0:
            decode_start_sec = max(0.0, chunk_start_sec - CHUNK_BACKSHIFT_SECONDS)

        chunk_start = int(decode_start_sec * sample_rate)
        chunk_end = int(chunk_end_sec * sample_rate)
        chunk_audio = audio_data[chunk_start:chunk_end]

        if len(chunk_audio) == 0:
            continue

        chunk_offset_sec = base_offset_sec + chunk_start_sec
        chunk_end_sec_absolute = base_offset_sec + chunk_end_sec
        decode_offset_sec = base_offset_sec + decode_start_sec
        print(
            f"[{speaker_tag}] Chunk {chunk_index}/{total_chunks} | decode {decode_offset_sec:.2f}s -> {chunk_end_sec_absolute:.2f}s | "
            f"segment {chunk_offset_sec:.2f}s -> {chunk_end_sec_absolute:.2f}s"
        )

        text = decode_chunk_text(
            chunk_audio=chunk_audio,
            sample_rate=sample_rate,
            processor=processor,
            model=model,
            device=device,
            lm_decoder=lm_decoder,
        )

        if not text:
            continue

        if segments:
            previous_text = segments[-1]["text"]
            text, overlap_words = trim_boundary_overlap(
                previous_text=previous_text,
                current_text=text,
                max_overlap_words=BOUNDARY_MAX_OVERLAP_WORDS,
            )
            if overlap_words > 0:
                print(f"  ↳ [{speaker_tag}] overlap trim: odstraněno {overlap_words} slov na hraně segmentu")

        if not text:
            continue

        segments.append(
            {
                "speaker": speaker_tag,
                "start": float(chunk_offset_sec),
                "end": float(chunk_end_sec_absolute),
                "text": text,
                "_norm": normalize_for_dedup(text),
            }
        )

    return segments


def merge_and_deduplicate(left_segments, right_segments):
    all_segments = sorted(left_segments + right_segments, key=lambda item: (item["start"], item["end"]))
    merged = []

    for segment in all_segments:
        if not segment.get("_norm"):
            continue

        is_duplicate = False
        for kept in reversed(merged):
            if segment["start"] - kept["start"] > DEDUP_TIME_TOLERANCE:
                break

            same_text = segment["_norm"] == kept.get("_norm")
            overlap = not (segment["end"] < kept["start"] or segment["start"] > kept["end"])
            if same_text and overlap and segment.get("speaker") != kept.get("speaker"):
                is_duplicate = True
                break

        if not is_duplicate:
            merged.append(segment)

    for segment in merged:
        segment.pop("_norm", None)

    return merged


def enrich_segment_schema(segments):
    enriched = []
    for index, segment in enumerate(segments, start=1):
        speaker_value = segment.get("speaker", "unknown")
        enriched.append(
            {
                "id": f"asr_{index:05d}",
                "speaker": speaker_value,
                "speakers": [speaker_value],
                "start": round(float(segment["start"]), 3),
                "end": round(float(segment["end"]), 3),
                "text": segment["text"],
                "is_overlap": False,
            }
        )
    return enriched


def resolve_individual_audio_path():
    if len(sys.argv) != 2 or not sys.argv[1].strip():
        print("Použití: python scripts/asr_individual_interspeech.py <cesta_k_stereo_wav>")
        sys.exit(1)

    return Path(sys.argv[1].strip()).expanduser().resolve()


def run_individual_asr_interspeech():
    run_started_utc = datetime.now(timezone.utc)
    runtime_start = time.perf_counter()
    stereo_audio_path = resolve_individual_audio_path()

    start_sec, end_sec = resolve_time_window(START_SECONDS, END_SECONDS)
    has_range = start_sec is not None or end_sec is not None
    output_scope = "full" if start_sec is None and end_sec is None else "range"
    output_name = f"{stereo_audio_path.stem}_{output_scope}_interspeech.json"
    output_path = PROJECT_ROOT / "results" / output_name

    if not os.path.exists(stereo_audio_path):
        print(f"Chyba: Soubor {stereo_audio_path} nebyl nalezen!")
        return

    if not DEFAULT_MODEL_DIR.exists():
        print(f"Chyba: Model složka neexistuje: {DEFAULT_MODEL_DIR}")
        return

    print(f"Načítám Interspeech W2V2 model z: {DEFAULT_MODEL_DIR}")
    processor = Wav2Vec2Processor.from_pretrained(str(DEFAULT_MODEL_DIR), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(DEFAULT_MODEL_DIR), local_files_only=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    lm_path = None
    lm_candidate = DEFAULT_MODEL_DIR / "LM.arpa"
    if lm_candidate.exists():
        lm_path = str(lm_candidate)

    lm_decoder, lm_enabled, lm_message = try_build_lm_decoder(processor, lm_path)
    print(lm_message)

    print("Načítám stereo audio a rozděluji kanály (L/R)...")
    try:
        left_audio, right_audio, sample_rate, base_offset_sec = load_stereo_audio_scipy(
            str(stereo_audio_path),
            start_sec,
            end_sec,
        )
    except ValueError as exc:
        print(f"Chyba: {exc}")
        return

    if len(left_audio) == 0 or len(right_audio) == 0:
        print("Chyba: Zvolený časový rozsah neobsahuje žádná data v jednom z kanálů.")
        return

    print("Zpracovávám LEVÝ kanál (Speaker_L)...")
    left_segments = transcribe_channel(
        processor=processor,
        model=model,
        device=device,
        audio_data=left_audio,
        sample_rate=sample_rate,
        speaker_tag="Speaker_L",
        base_offset_sec=base_offset_sec,
        lm_decoder=lm_decoder,
    )

    print("Zpracovávám PRAVÝ kanál (Speaker_R)...")
    right_segments = transcribe_channel(
        processor=processor,
        model=model,
        device=device,
        audio_data=right_audio,
        sample_rate=sample_rate,
        speaker_tag="Speaker_R",
        base_offset_sec=base_offset_sec,
        lm_decoder=lm_decoder,
    )

    print("Slučuji segmenty a odstraňuji duplicity mezi kanály...")
    merged_segments = merge_and_deduplicate(left_segments, right_segments)
    enriched_segments = enrich_segment_schema(merged_segments)

    speaker_l_full_transcription = " ".join(segment["text"] for segment in left_segments)
    speaker_r_full_transcription = " ".join(segment["text"] for segment in right_segments)
    full_transcription = " ".join(segment["text"] for segment in enriched_segments)

    runtime_seconds = round(time.perf_counter() - runtime_start, 2)
    run_finished_utc = datetime.now(timezone.utc)

    final_output = {
        "metadata": {
            "mode": "INDIVIDUAL_SPLIT",
            "start_seconds": start_sec,
            "end_seconds": end_sec,
            "has_custom_range": has_range,
            "model": DEFAULT_MODEL_DIR.name,
            "backend": "interspeech",
            "dedup_time_tolerance": DEDUP_TIME_TOLERANCE,
            "chunk_seconds": CHUNK_SECONDS,
            "chunk_backshift_seconds": CHUNK_BACKSHIFT_SECONDS,
            "boundary_max_overlap_words": BOUNDARY_MAX_OVERLAP_WORDS,
            "vad_min_dur": VAD_MIN_DUR,
            "vad_max_dur": VAD_MAX_DUR,
            "vad_max_silence": VAD_MAX_SILENCE,
            "vad_energy_threshold": VAD_ENERGY_THRESHOLD,
            "lm_enabled": lm_enabled,
            "lm_path": lm_path,
            "runtime_seconds": runtime_seconds,
            "run_started_utc": run_started_utc.isoformat(),
            "run_finished_utc": run_finished_utc.isoformat(),
        },
        "segments": enriched_segments,
        "Speaker_L_full_transcription": speaker_l_full_transcription,
        "speaker_R_full_transcription": speaker_r_full_transcription,
        "full_transcription": full_transcription,
    }

    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(final_output, output_file, ensure_ascii=False, indent=2)

    print(f"Transkripce hotova a uložena do: {output_path}")


if __name__ == "__main__":
    run_individual_asr_interspeech()
