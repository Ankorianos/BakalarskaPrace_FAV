import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import whisper
from scipy.io import wavfile

START_SECONDS = 360
END_SECONDS = 540
MODEL_SIZE = "turbo"
CHUNK_SECONDS = 25
CHUNK_OVERLAP_SECONDS = 2.0
ADJACENT_MERGE_GAP_SECONDS = 1.2
SHORT_CONTINUATION_WORDS = 4
MAX_ADJACENT_MERGED_WORDS = 36

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
                "start": segment["start"],
                "end": segment["end"],
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


def transcribe_in_chunks(model, audio_data, sample_rate, base_offset_sec=0.0):
    total_samples = len(audio_data)
    chunk_samples = int(CHUNK_SECONDS * sample_rate)
    overlap_samples = int(CHUNK_OVERLAP_SECONDS * sample_rate)
    step_samples = max(1, chunk_samples - overlap_samples)

    if chunk_samples <= 0:
        return []

    segments = []
    total_chunks = max(1, (total_samples + step_samples - 1) // step_samples)
    chunk_index = 0

    for chunk_start in range(0, total_samples, step_samples):
        chunk_index += 1
        chunk_end = min(chunk_start + chunk_samples, total_samples)
        chunk_audio = audio_data[chunk_start:chunk_end]

        if len(chunk_audio) == 0:
            continue

        chunk_offset_sec = base_offset_sec + (chunk_start / sample_rate)
        chunk_end_sec = base_offset_sec + (chunk_end / sample_rate)
        print(f"Chunk {chunk_index}/{total_chunks} | {chunk_offset_sec:.2f}s -> {chunk_end_sec:.2f}s")
        is_first_chunk = chunk_start == 0
        is_last_chunk = chunk_end >= total_samples
        keep_from = chunk_offset_sec if is_first_chunk else chunk_offset_sec + (CHUNK_OVERLAP_SECONDS / 2.0)
        keep_to = chunk_end_sec if is_last_chunk else chunk_end_sec - (CHUNK_OVERLAP_SECONDS / 2.0)

        result = model.transcribe(
            chunk_audio,
            language="cs",
            verbose=False,
            condition_on_previous_text=False,
            no_speech_threshold=0.25,
            temperature=0.0,
        )

        for segment in result.get("segments", []):
            text = clean_segment_text(segment.get("text", ""))
            if not text:
                continue

            absolute_start = float(segment["start"]) + chunk_offset_sec
            absolute_end = float(segment["end"]) + chunk_offset_sec
            segment_mid = (absolute_start + absolute_end) / 2.0

            if segment_mid < keep_from or segment_mid > keep_to:
                continue

            segments.append(
                {
                    "speakers": ["unknown"],
                    "start": round(absolute_start, 3),
                    "end": round(absolute_end, 3),
                    "text": text,
                }
            )

        if chunk_end >= total_samples:
            break

    deduped = deduplicate_segments(segments, time_tolerance=1.5)
    merged_overlaps = merge_overlapping_segments(deduped)
    boundary_cleaned = strip_boundary_artifacts(merged_overlaps)
    return merge_adjacent_segments(boundary_cleaned)


def run_mono_asr():
    run_started_utc = datetime.now(timezone.utc)
    runtime_start = time.perf_counter()

    audio_path = PROJECT_ROOT / "data" / "12008_001_MONO.wav"

    start_sec, end_sec = resolve_time_window(START_SECONDS, END_SECONDS)

    has_range = start_sec is not None or end_sec is not None
    if start_sec is None and end_sec is None:
        output_name = "mono_results_full.json"
    else:
        output_name = "mono_results_range.json"

    output_path = PROJECT_ROOT / "results" / output_name

    if not os.path.exists(audio_path):
        print(f"Chyba: Soubor {audio_path} nebyl nalezen!")
        return

    print(f"Načítám Whisper model ({MODEL_SIZE})...")
    model = whisper.load_model(MODEL_SIZE)

    print(f"Načítám MONO audio (start={start_sec}, end={end_sec})...")
    audio_data, sample_rate, base_offset_sec = load_audio_segment(str(audio_path), start_sec, end_sec)

    if len(audio_data) == 0:
        print("Chyba: Zvolený časový rozsah neobsahuje žádná data.")
        return

    print("Spouštím MONO rozpoznávání...")
    asr_segments = transcribe_in_chunks(model, audio_data, sample_rate, base_offset_sec=base_offset_sec)
    asr_segments = enrich_segment_schema(asr_segments)

    full_transcription = " ".join(segment["text"] for segment in asr_segments)
    runtime_seconds = round(time.perf_counter() - runtime_start, 2)
    run_finished_utc = datetime.now(timezone.utc)

    final_output = {
        "metadata": {
            "mode": "MONO",
            "start_seconds": start_sec,
            "end_seconds": end_sec,
            "has_custom_range": has_range,
            "model": MODEL_SIZE,
            "backend": "whisper",
            "chunk_seconds": CHUNK_SECONDS,
            "chunk_overlap_seconds": CHUNK_OVERLAP_SECONDS,
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
    run_mono_asr()
