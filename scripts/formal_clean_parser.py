import json
import re
from collections import defaultdict
from pathlib import Path

EVAL_SPEAKERS = ["interviewer", "interviewee"]
SPEAKER_CODE_MAP = {
    "R": "interviewer",
    "M": "interviewee",
    "F": "interviewee",
}

LINE_ID_PATTERN = re.compile(
    r"^(?P<recording>\d+)_(?P<speaker_code>[MFR])_(?P<part>\d+)_\((?P<time>\d+\.\d+)\)$"
)


def clean_text_for_asr(text):
    if not text:
        return ""
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_text_with_fallback(file_path):
    encodings = ["utf-8", "windows-1250", "cp1250", "latin-2"]
    last_error = None

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as file_handle:
                return file_handle.read(), encoding
        except UnicodeDecodeError as error:
            last_error = error

    raise last_error or UnicodeDecodeError("unknown", b"", 0, 1, "Nepodařilo se načíst soubor")


def parse_formal_lines(text):
    parsed_rows = []

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        if "\t" not in line:
            continue

        line_id, text_part = line.split("\t", 1)
        text_cleaned = clean_text_for_asr(text_part)
        if not text_cleaned:
            continue

        match = LINE_ID_PATTERN.match(line_id.strip())
        if not match:
            continue

        recording = match.group("recording")
        speaker_code = match.group("speaker_code")
        part = int(match.group("part"))
        rel_time = float(match.group("time"))

        parsed_rows.append(
            {
                "line_number": line_number,
                "recording": recording,
                "recording_key": f"{recording}_{part:03d}",
                "speaker_code": speaker_code,
                "speaker": SPEAKER_CODE_MAP.get(speaker_code, "unknown"),
                "part": part,
                "relative_start": rel_time,
                "text": text_cleaned,
            }
        )

    return parsed_rows


def build_part_offsets(rows_for_recording):
    max_time_by_part = defaultdict(float)

    for row in rows_for_recording:
        part = row["part"]
        max_time_by_part[part] = max(max_time_by_part[part], row["relative_start"])

    offsets = {}
    cumulative = 0.0
    for part in sorted(max_time_by_part):
        offsets[part] = cumulative
        cumulative += max_time_by_part[part]

    return offsets


def compute_default_segment_duration(starts):
    if len(starts) < 2:
        return 3.0

    gaps = []
    for index in range(len(starts) - 1):
        gap = starts[index + 1] - starts[index]
        if gap > 0:
            gaps.append(gap)

    if not gaps:
        return 3.0

    gaps_sorted = sorted(gaps)
    mid = len(gaps_sorted) // 2
    if len(gaps_sorted) % 2 == 1:
        median = gaps_sorted[mid]
    else:
        median = 0.5 * (gaps_sorted[mid - 1] + gaps_sorted[mid])

    return min(10.0, max(0.5, median))


def build_eval_segments_by_recording(parsed_rows):
    rows_by_recording = defaultdict(list)
    for row in parsed_rows:
        rows_by_recording[row["recording_key"]].append(row)

    eval_by_recording = {}

    def sort_key(value):
        match = re.match(r"^(\d+)_(\d+)$", str(value))
        if not match:
            return (10**9, 10**9)
        return (int(match.group(1)), int(match.group(2)))

    for recording in sorted(rows_by_recording.keys(), key=sort_key):
        rows = rows_by_recording[recording]

        normalized = sorted(rows, key=lambda row: (row["relative_start"], row["line_number"]))

        starts = [row["relative_start"] for row in normalized]
        fallback_duration = compute_default_segment_duration(starts)

        recording_segments = []

        for index, row in enumerate(normalized, start=1):
            start_time = row["relative_start"]

            if index < len(normalized):
                next_start = normalized[index]["relative_start"]
                end_time = max(start_time + 0.001, next_start)
            else:
                end_time = start_time + fallback_duration

            recording_segments.append(
                {
                    "id": f"seg_{recording}_{index:05d}",
                    "speakers": [row["speaker"]],
                    "start": round(float(start_time), 3),
                    "end": round(float(end_time), 3),
                    "text": row["text"],
                    "is_overlap": False,
                }
            )

        eval_by_recording[recording] = recording_segments

    return eval_by_recording


def build_full_speakers_summary_by_recording(eval_by_recording):
    summary_by_recording = {}

    def sort_key(value):
        match = re.match(r"^(\d+)_(\d+)$", str(value))
        if not match:
            return (10**9, 10**9)
        return (int(match.group(1)), int(match.group(2)))

    for recording in sorted(eval_by_recording.keys(), key=sort_key):
        recording_segments = eval_by_recording[recording]
        min_start = min(segment["start"] for segment in recording_segments)
        max_end = max(segment["end"] for segment in recording_segments)

        speaker_map = {}

        for speaker in EVAL_SPEAKERS:
            text_parts = [
                clean_text_for_asr(segment.get("text", ""))
                for segment in recording_segments
                if segment.get("speakers") == [speaker]
            ]
            text_parts = [text for text in text_parts if text]
            full_text = " ".join(text_parts).strip()

            if not full_text:
                continue

            speaker_map[speaker] = {
                "id": f"full_{recording}_{speaker}",
                "speakers": [speaker],
                "start": round(float(min_start), 3),
                "end": round(float(max_end), 3),
                "text": full_text,
                "is_overlap": False,
            }

        if speaker_map:
            summary_by_recording[recording] = speaker_map

    return summary_by_recording


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, ensure_ascii=False, indent=2)


def main():
    project_root = Path(__file__).resolve().parents[1]
    input_file = project_root / "data" / "FormalForEvaluation.txt"

    eval_output = project_root / "results" / "formal_ground_truth_eval.json"
    speakers_output = project_root / "results" / "formal_ground_truth_speakers.json"

    if not input_file.exists():
        raise FileNotFoundError(f"Vstupní soubor nebyl nalezen: {input_file}")

    text, used_encoding = read_text_with_fallback(input_file)
    parsed_rows = parse_formal_lines(text)
    eval_by_recording = build_eval_segments_by_recording(parsed_rows)
    speakers_by_recording = build_full_speakers_summary_by_recording(eval_by_recording)

    eval_segments_count = sum(len(segments) for segments in eval_by_recording.values())
    speakers_segments_count = sum(len(speakers) for speakers in speakers_by_recording.values())

    save_json(eval_output, eval_by_recording)
    save_json(speakers_output, speakers_by_recording)

    print(f"Načten encoding: {used_encoding}")
    print(f"Vstup: {input_file}")
    print(f"EVAL nahrávek: {len(eval_by_recording)}")
    print(f"EVAL segmentů: {eval_segments_count}")
    print(f"SPEAKERS nahrávek: {len(speakers_by_recording)}")
    print(f"SPEAKERS segmentů: {speakers_segments_count}")
    print(f"EVAL uložen do: {eval_output}")
    print(f"SPEAKERS uložen do: {speakers_output}")


if __name__ == "__main__":
    main()
