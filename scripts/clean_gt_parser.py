import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

EVAL_SPEAKERS = ["interviewer", "interviewee"]


def clean_text_for_asr(text):
    if not text:
        return ""
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_speaker_name(speaker_id, speaker_map):
    return speaker_map.get(speaker_id, speaker_id or "unknown")


def normalize_for_compare(text):
    normalized = (text or "").lower()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return " ".join(normalized.split()).strip()


def add_text_bucket(text_buckets, speaker_name, text):
    cleaned = clean_text_for_asr(text)
    if not cleaned:
        return
    text_buckets[speaker_name] = f"{text_buckets.get(speaker_name, '')} {cleaned}".strip()


def make_raw_segment(start_time, end_time, text_buckets, global_index):
    if end_time < start_time:
        return None

    speakers_in_segment = [speaker for speaker, text in text_buckets.items() if clean_text_for_asr(text)]
    if not speakers_in_segment:
        return None

    ordered_speakers = sorted(speakers_in_segment, key=lambda name: EVAL_SPEAKERS.index(name) if name in EVAL_SPEAKERS else 99)
    is_overlap = len(ordered_speakers) > 1

    combined_text = " ".join(text_buckets[speaker] for speaker in ordered_speakers).strip()

    segment = {
        "id": f"seg_{global_index:05d}",
        "speakers": ordered_speakers,
        "start": round(float(start_time), 3),
        "end": round(float(end_time), 3),
        "text": combined_text,
        "is_overlap": is_overlap,
    }

    if is_overlap:
        segment["text_by_speaker"] = {speaker: text_buckets[speaker] for speaker in ordered_speakers}

    return segment


def extract_recording_id_from_path(file_path):
    stem = Path(file_path).stem
    match = re.match(r"^(\d+_\d+)", stem)
    if not match:
        match = re.match(r"^(\d+)", stem)
    return match.group(1) if match else "unknown"


def parse_turn(turn, speaker_map, start_index):
    turn_start = float(turn.get("startTime"))
    turn_end = float(turn.get("endTime"))

    speaker_ids = (turn.get("speaker") or "").split()
    resolved_speakers = [normalize_speaker_name(speaker_id, speaker_map) for speaker_id in speaker_ids]

    segments = []
    current_start = turn_start
    current_speaker_index = 0 if len(resolved_speakers) == 1 else None
    text_buckets = {}
    next_index = start_index

    def current_speaker_name():
        if current_speaker_index is None:
            return "unknown"
        if current_speaker_index < len(resolved_speakers):
            return resolved_speakers[current_speaker_index]
        return resolved_speakers[-1] if resolved_speakers else "unknown"

    def add_current_text(text):
        add_text_bucket(text_buckets, current_speaker_name(), text)

    add_current_text(turn.text)

    for child in turn:
        tag = child.tag

        if tag == "Sync":
            sync_time = float(child.get("time"))
            segment = make_raw_segment(
                current_start,
                sync_time,
                text_buckets,
                next_index,
            )
            if segment:
                segments.append(segment)
                next_index += 1

            text_buckets = {}
            current_start = sync_time
            current_speaker_index = 0 if len(resolved_speakers) == 1 else None

            add_current_text(child.tail)
            continue

        if tag == "Who":
            who_index = int(child.get("nb", "1")) - 1
            current_speaker_index = max(0, who_index)
            add_current_text(child.tail)
            continue

        add_current_text(child.text)
        add_current_text(child.tail)

    segment = make_raw_segment(
        current_start,
        turn_end,
        text_buckets,
        next_index,
    )
    if segment:
        segments.append(segment)
        next_index += 1

    return segments, next_index


def parse_ground_truth_raw(file_path):
    with open(file_path, "r", encoding="windows-1250") as file:
        xml_data = file.read()

    xml_data = re.sub(r"<!DOCTYPE.*?>", "", xml_data)
    root = ET.fromstring(xml_data)

    speaker_map = {}
    for speaker in root.findall(".//Speaker"):
        speaker_map[speaker.get("id")] = speaker.get("name")

    raw_segments = []
    next_index = 1
    for turn in root.findall(".//Turn"):
        turn_segments, next_index = parse_turn(turn, speaker_map, next_index)
        raw_segments.extend(turn_segments)

    return raw_segments


def build_eval_reference(raw_segments, recording_id=None):
    eval_segments = []

    for raw_segment in raw_segments:
        selected_speakers = [speaker for speaker in raw_segment.get("speakers", []) if speaker in EVAL_SPEAKERS]
        if not selected_speakers:
            continue

        if raw_segment.get("is_overlap", False) and "text_by_speaker" in raw_segment:
            text_parts = []
            text_by_speaker = {}
            for speaker in EVAL_SPEAKERS:
                if speaker in selected_speakers and speaker in raw_segment["text_by_speaker"]:
                    speaker_text = clean_text_for_asr(raw_segment["text_by_speaker"][speaker])
                    if speaker_text:
                        text_parts.append(speaker_text)
                        text_by_speaker[speaker] = speaker_text

            if not text_parts:
                continue

            eval_text = " ".join(text_parts).strip()
            eval_is_overlap = len(text_by_speaker) > 1
            eval_speakers = list(text_by_speaker.keys())
        else:
            eval_text = clean_text_for_asr(raw_segment.get("text", ""))
            if not eval_text:
                continue
            eval_speakers = selected_speakers
            eval_is_overlap = len(eval_speakers) > 1
            text_by_speaker = None

        raw_id = raw_segment.get("id")
        if recording_id and isinstance(raw_id, str) and raw_id.startswith("seg_"):
            segment_id = raw_id.replace("seg_", f"seg_{recording_id}_", 1)
        else:
            segment_id = raw_id

        segment = {
            "id": segment_id,
            "speakers": eval_speakers,
            "start": raw_segment["start"],
            "end": raw_segment["end"],
            "text": eval_text,
            "is_overlap": eval_is_overlap,
        }

        if eval_is_overlap and text_by_speaker:
            segment["text_by_speaker"] = text_by_speaker

        eval_segments.append(segment)

    # Dedup přesné duplicity (občas vzniknou při složitých Who strukturách)
    deduped = []
    seen = set()
    for segment in eval_segments:
        key = (
            segment["start"],
            segment["end"],
            tuple(segment.get("speakers", [])),
            normalize_for_compare(segment["text"]),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(segment)

    return deduped


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def build_full_speakers_summary(segments, recording_id=None):
    if not segments:
        return []

    min_start = min(segment["start"] for segment in segments)
    max_end = max(segment["end"] for segment in segments)

    summary_segments = []
    for speaker_name in EVAL_SPEAKERS:
        text_parts = []

        for segment in segments:
            if segment.get("is_overlap") and "text_by_speaker" in segment:
                speaker_text = clean_text_for_asr(segment["text_by_speaker"].get(speaker_name, ""))
                if speaker_text:
                    text_parts.append(speaker_text)
                continue

            if segment.get("speakers") == [speaker_name]:
                speaker_text = clean_text_for_asr(segment.get("text", ""))
                if speaker_text:
                    text_parts.append(speaker_text)

        full_text = " ".join(text_parts).strip()
        if not full_text:
            continue

        summary_segments.append(
            {
                "id": f"full_{recording_id}_{speaker_name}" if recording_id else f"full_{speaker_name}",
                "speakers": [speaker_name],
                "start": round(float(min_start), 3),
                "end": round(float(max_end), 3),
                "text": full_text,
                "is_overlap": False,
            }
        )

    return summary_segments


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    trs_file = project_root / "data" / "12008_001.trs"

    raw_output = project_root / "results" / "ground_truth_raw.json"
    eval_output = project_root / "results" / "ground_truth_eval.json"
    speakers_output = project_root / "results" / "ground_truth_speakers.json"

    recording_id = extract_recording_id_from_path(str(trs_file))

    raw_segments = parse_ground_truth_raw(str(trs_file))
    eval_segments = build_eval_reference(raw_segments, recording_id=recording_id)
    full_speakers = build_full_speakers_summary(eval_segments, recording_id=recording_id)

    raw_output_data = {recording_id: raw_segments}
    eval_output_data = {recording_id: eval_segments}
    speakers_output_data = {
        recording_id: {item["speakers"][0]: item for item in full_speakers}
    }

    save_json(raw_output, raw_output_data)
    save_json(eval_output, eval_output_data)
    save_json(speakers_output, speakers_output_data)

    print(f"Hotovo! RAW segmentů: {len(raw_segments)}")
    print(f"Hotovo! EVAL segmentů: {len(eval_segments)}")
    print(f"Hotovo! SPEAKERS segmentů: {len(full_speakers)}")
    print(f"Recording ID: {recording_id}")
    print(f"RAW uložen do: {raw_output}")
    print(f"EVAL uložen do: {eval_output}")
    print(f"SPEAKERS uložen do: {speakers_output}")

    print("\nUkázka EVAL (prvních 10):")
    for segment in eval_segments[:10]:
        print(f"[{segment['start']:>7.2f}] {' + '.join(segment['speakers']):<28}: {segment['text']}")
