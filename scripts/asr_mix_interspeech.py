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
CHUNK_SECONDS = 360
CHUNK_OVERLAP_SECONDS = 2.0
ADJACENT_MERGE_GAP_SECONDS = 1.2
SHORT_CONTINUATION_WORDS = 4
MAX_ADJACENT_MERGED_WORDS = 36
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


def normalize_word_for_match(word):
    normalized = re.sub(r"[^\w]", "", (word or "").lower())
    return normalized


def is_text_contained(short_text, long_text):
    short_norm = normalize_for_dedup(short_text)
    long_norm = normalize_for_dedup(long_text)
    if not short_norm or not long_norm:
        return False
    return short_norm in long_norm


def overlap_token_count(left_text, right_text, max_window=12):
    left_tokens = normalize_for_dedup(left_text).split()
    right_tokens = normalize_for_dedup(right_text).split()
    if not left_tokens or not right_tokens:
        return 0

    max_k = min(len(left_tokens), len(right_tokens), max_window)
    for token_count in range(max_k, 0, -1):
        if left_tokens[-token_count:] == right_tokens[:token_count]:
            return token_count
    return 0


def ends_with_spelled_letters(text):
    return re.search(r"(?:\b[\wÁČĎÉĚÍŇÓŘŠŤÚŮÝŽáčďéěíňóřšťúůýž]-){2,}[\wÁČĎÉĚÍŇÓŘŠŤÚŮÝŽáčďéěíňóřšťúůýž]$", text or "") is not None


def strip_boundary_artifacts(segments):
    cleaned_segments = [dict(segment) for segment in segments]

    for index in range(1, len(cleaned_segments)):
        previous_text = cleaned_segments[index - 1]["text"]
        current_text = cleaned_segments[index]["text"]

        if not ends_with_spelled_letters(previous_text):
            continue

        match = re.match(
            r"^([A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ]{3,})([,.]?)\s*(.*)$",
            current_text,
        )
        if not match:
            continue

        trailing_text = (match.group(3) or "").strip()
        if trailing_text:
            cleaned_segments[index]["text"] = trailing_text

    return cleaned_segments


def should_merge_adjacent(previous, current):
    gap = current["start"] - previous["end"]
    if gap > ADJACENT_MERGE_GAP_SECONDS:
        return False

    prev_text = previous["text"].strip()
    curr_text = current["text"].strip()
    if not prev_text or not curr_text:
        return False

    curr_starts_lower = re.match(r"^[a-záčďéěíňóřšťúůýž]", curr_text) is not None
    curr_is_short = len(curr_text.split()) <= SHORT_CONTINUATION_WORDS
    token_overlap = overlap_token_count(prev_text, curr_text)
    prev_is_question = prev_text.endswith("?")
    has_strong_text_overlap = token_overlap >= 2
    has_text_containment = is_text_contained(curr_text, prev_text) or is_text_contained(prev_text, curr_text)
    would_be_too_long = len((prev_text + " " + curr_text).split()) > MAX_ADJACENT_MERGED_WORDS

    if would_be_too_long:
        return False

    if prev_is_question and not curr_starts_lower and not curr_is_short:
        return False

    return curr_starts_lower or curr_is_short or has_strong_text_overlap or has_text_containment


def merge_adjacent_segments(segments):
    if not segments:
        return []

    merged = [dict(segments[0])]

    def replace_repeated_tail(previous_text, current_text):
        previous_words = previous_text.split()
        current_words = current_text.split()

        if len(previous_words) < 3 or len(current_words) < 2 or len(current_words) > 8:
            return None

        previous_norm = [normalize_word_for_match(word) for word in previous_words]
        current_norm = [normalize_word_for_match(word) for word in current_words]

        max_prefix = min(4, len(current_norm), len(previous_norm))
        start_window = max(0, len(previous_norm) - 8)

        for prefix_len in range(max_prefix, 1, -1):
            prefix = current_norm[:prefix_len]

            for index in range(len(previous_norm) - prefix_len, start_window - 1, -1):
                if previous_norm[index : index + prefix_len] == prefix:
                    merged_words = previous_words[:index] + current_words
                    return " ".join(merged_words).strip()

        return None

    for segment in segments[1:]:
        previous = merged[-1]

        if not should_merge_adjacent(previous, segment):
            merged.append(dict(segment))
            continue

        token_overlap = overlap_token_count(previous["text"], segment["text"])
        previous_is_question = previous["text"].strip().endswith("?")

        if token_overlap > 0 and not previous_is_question:
            words = segment["text"].split()
            suffix = " ".join(words[token_overlap:]).strip()
            if suffix:
                previous["text"] = f"{previous['text']} {suffix}".strip()
        elif is_text_contained(segment["text"], previous["text"]):
            pass
        elif is_text_contained(previous["text"], segment["text"]):
            previous["text"] = segment["text"]
        else:
            tail_replaced = replace_repeated_tail(previous["text"], segment["text"])
            if tail_replaced:
                previous["text"] = tail_replaced
            else:
                previous["text"] = f"{previous['text']} {segment['text']}".strip()

        previous["end"] = max(previous["end"], segment["end"])

    return merged


def merge_overlapping_segments(segments):
    if not segments:
        return []

    merged = [dict(segments[0])]

    for segment in segments[1:]:
        previous = merged[-1]
        temporal_overlap = segment["start"] <= previous["end"]

        if not temporal_overlap:
            merged.append(dict(segment))
            continue

        token_overlap = overlap_token_count(previous["text"], segment["text"])
        if token_overlap > 0:
            segment_words = segment["text"].split()
            suffix = " ".join(segment_words[token_overlap:]).strip()
            if suffix:
                previous["text"] = f"{previous['text']} {suffix}".strip()
            previous["end"] = max(previous["end"], segment["end"])
            continue

        if is_text_contained(segment["text"], previous["text"]):
            previous["end"] = max(previous["end"], segment["end"])
            continue

        if is_text_contained(previous["text"], segment["text"]):
            previous["text"] = segment["text"]
            previous["start"] = min(previous["start"], segment["start"])
            previous["end"] = max(previous["end"], segment["end"])
            continue

        segment_copy = dict(segment)
        segment_copy["start"] = round(previous["end"] + 0.001, 3)
        if segment_copy["start"] >= segment_copy["end"]:
            segment_copy["end"] = round(segment_copy["start"] + 0.001, 3)
        merged.append(segment_copy)

    return merged


def deduplicate_segments(segments, time_tolerance=0.8):
    deduped = []

    for segment in sorted(segments, key=lambda item: (item["start"], item["end"])):
        norm_text = normalize_for_dedup(segment["text"])
        if not norm_text:
            continue

        duplicate = False
        for previous in reversed(deduped):
            if segment["start"] - previous["start"] > time_tolerance:
                break

            same_text = norm_text == normalize_for_dedup(previous["text"])
            contained_in_previous = is_text_contained(segment["text"], previous["text"])
            contains_previous = is_text_contained(previous["text"], segment["text"])
            overlap = not (segment["end"] <= previous["start"] or segment["start"] >= previous["end"])
            if (same_text or contained_in_previous) and overlap:
                duplicate = True
                break
            if contains_previous and overlap:
                previous["start"] = min(previous["start"], segment["start"])
                previous["end"] = max(previous["end"], segment["end"])
                previous["text"] = segment["text"]
                duplicate = True
                break

        if not duplicate:
            deduped.append(segment)

    return deduped


def enrich_segment_schema(segments):
    enriched = []
    for index, segment in enumerate(segments, start=1):
        speakers = segment.get("speakers")
        if not isinstance(speakers, list) or not speakers:
            speakers = ["unknown"]

        enriched.append(
            {
                "id": f"asr_{index:05d}",
                "speakers": speakers,
                "start": round(float(segment["start"]), 3),
                "end": round(float(segment["end"]), 3),
                "text": segment["text"],
                "is_overlap": False,
            }
        )
    return enriched


def resolve_time_window(start_sec=None, end_sec=None):
    if start_sec is not None and start_sec < 0:
        raise ValueError("START_SECONDS musí být >= 0 nebo None")
    if end_sec is not None and end_sec < 0:
        raise ValueError("END_SECONDS musí být >= 0 nebo None")
    if start_sec is not None and end_sec is not None and end_sec <= start_sec:
        raise ValueError("END_SECONDS musí být větší než START_SECONDS")

    return start_sec, end_sec


def load_audio_segment(file_path, start_sec=None, end_sec=None):
    sample_rate, data = wavfile.read(file_path)

    if len(data.shape) > 1:
        data = data[:, 0]

    start_sec, end_sec = resolve_time_window(start_sec, end_sec)

    start_sample = int(start_sec * sample_rate) if start_sec is not None else 0
    end_sample = int(end_sec * sample_rate) if end_sec is not None else len(data)
    end_sample = min(end_sample, len(data))

    if start_sample >= len(data):
        data = np.array([], dtype=np.float32)
    else:
        data = data[start_sample:end_sample]

    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)

    return data, sample_rate, (start_sample / sample_rate)


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


def transcribe_in_chunks(processor, model, device, audio_data, sample_rate, base_offset_sec=0.0, lm_decoder=None):
    segments = []
    audio_duration = len(audio_data) / float(sample_rate)

    speech_windows = detect_speech_windows_with_auditok(audio_data, sample_rate)
    vad_chunks = build_vad_chunks(speech_windows, audio_duration, CHUNK_SECONDS)
    total_chunks = len(vad_chunks)

    for chunk_index, (chunk_start_sec, chunk_end_sec) in enumerate(vad_chunks, start=1):
        chunk_start = int(chunk_start_sec * sample_rate)
        chunk_end = int(chunk_end_sec * sample_rate)
        chunk_audio = audio_data[chunk_start:chunk_end]

        if len(chunk_audio) == 0:
            continue

        chunk_offset_sec = base_offset_sec + chunk_start_sec
        chunk_end_sec_absolute = base_offset_sec + chunk_end_sec
        print(f"Chunk {chunk_index}/{total_chunks} | {chunk_offset_sec:.2f}s -> {chunk_end_sec_absolute:.2f}s")

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

        segments.append(
            {
                "speakers": ["unknown"],
                "start": float(chunk_offset_sec),
                "end": float(chunk_end_sec_absolute),
                "text": text,
            }
        )

    deduped = deduplicate_segments(segments, time_tolerance=1.5)
    merged_overlaps = merge_overlapping_segments(deduped)
    boundary_cleaned = strip_boundary_artifacts(merged_overlaps)
    return merge_adjacent_segments(boundary_cleaned)


def resolve_mix_audio_path():
    if len(sys.argv) != 2 or not sys.argv[1].strip():
        print("Použití: python scripts/asr_mix_interspeech.py <cesta_k_mix_wav>")
        sys.exit(1)

    return Path(sys.argv[1].strip()).expanduser().resolve()


def run_mix_asr_interspeech():
    run_started_utc = datetime.now(timezone.utc)
    runtime_start = time.perf_counter()
    audio_path = resolve_mix_audio_path()

    start_sec, end_sec = resolve_time_window(START_SECONDS, END_SECONDS)

    has_range = start_sec is not None or end_sec is not None
    output_scope = "full" if start_sec is None and end_sec is None else "range"
    output_name = f"{audio_path.stem}_{output_scope}_interspeech.json"

    output_path = PROJECT_ROOT / "results" / output_name

    if not os.path.exists(audio_path):
        print(f"Chyba: Soubor {audio_path} nebyl nalezen!")
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

    print(f"Načítám MIX audio (start={start_sec}, end={end_sec})...")
    audio_data, sample_rate, base_offset_sec = load_audio_segment(str(audio_path), start_sec, end_sec)

    if len(audio_data) == 0:
        print("Chyba: Zvolený časový rozsah neobsahuje žádná data.")
        return

    print("Spouštím MIX rozpoznávání (Interspeech)...")
    asr_segments = transcribe_in_chunks(
        processor=processor,
        model=model,
        device=device,
        audio_data=audio_data,
        sample_rate=sample_rate,
        base_offset_sec=base_offset_sec,
        lm_decoder=lm_decoder,
    )
    asr_segments = enrich_segment_schema(asr_segments)

    full_transcription = " ".join(segment["text"] for segment in asr_segments)
    runtime_seconds = round(time.perf_counter() - runtime_start, 2)
    run_finished_utc = datetime.now(timezone.utc)

    final_output = {
        "metadata": {
            "mode": "MIX",
            "start_seconds": start_sec,
            "end_seconds": end_sec,
            "has_custom_range": has_range,
            "model": DEFAULT_MODEL_DIR.name,
            "backend": "interspeech",
            "chunk_seconds": CHUNK_SECONDS,
            "chunk_overlap_seconds": CHUNK_OVERLAP_SECONDS,
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
        "segments": asr_segments,
        "full_transcription": full_transcription,
    }

    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(final_output, output_file, ensure_ascii=False, indent=2)

    print(f"Hotovo. Segmenty uloženy do: {output_path}")


if __name__ == "__main__":
    run_mix_asr_interspeech()