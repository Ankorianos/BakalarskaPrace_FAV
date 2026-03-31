import argparse
import importlib
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "results" / "mono_results_range.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "mono_diarization_results_range.json"
DEFAULT_AUDIO = PROJECT_ROOT / "data" / "12008_001_MONO.wav"
DEFAULT_PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"

ANSWER_STARTERS = {
    "ano", "ne", "jo", "no", "to", "tak", "já", "my", "on", "ona", "ono",
    "maminka", "otec", "tatínek", "dědeček", "babička",
}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, ensure_ascii=False, indent=2)


def should_switch_speaker(previous_text: str, current_text: str):
    prev = (previous_text or "").strip().lower()
    curr = (current_text or "").strip().lower()

    if not curr:
        return False

    first_word = curr.split()[0] if curr.split() else ""
    if prev.endswith("?"):
        return True
    if first_word in ANSWER_STARTERS and len(curr.split()) >= 3:
        return True
    return False


def diarize_heuristic(segments, gap_switch_seconds=1.5):
    sorted_segments = sorted(
        [segment for segment in segments if isinstance(segment, dict)],
        key=lambda segment: float(segment.get("start", 0.0)),
    )

    speaker_labels = ["interviewer", "interviewee"]
    speaker_index = 0
    diarized = []
    previous_segment = None

    for segment in sorted_segments:
        start = float(segment.get("start", 0.0))
        text = str(segment.get("text", ""))

        if previous_segment is not None:
            previous_end = float(previous_segment.get("end", start))
            gap = max(0.0, start - previous_end)

            if gap >= gap_switch_seconds or should_switch_speaker(previous_segment.get("text", ""), text):
                speaker_index = 1 - speaker_index

        speaker = speaker_labels[speaker_index]
        segment_copy = dict(segment)
        segment_copy["speaker"] = speaker
        segment_copy["speakers"] = [speaker]
        diarized.append(segment_copy)
        previous_segment = segment_copy

    return diarized


def run_pyannote(audio_path: Path, model_name: str, hf_token: str, device: str):
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio soubor neexistuje: {audio_path}")

    if not hf_token:
        raise ValueError(
            "Chybí Hugging Face token. Použij --hf-token nebo nastav HF_TOKEN/HUGGINGFACE_TOKEN."
        )

    try:
        import torch
    except ImportError as import_error:
        raise ImportError(
            "Pro modelovou diarizaci nainstaluj: pip install pyannote.audio torch"
        ) from import_error

    try:
        pyannote_audio_module = importlib.import_module("pyannote.audio")
        Pipeline = getattr(pyannote_audio_module, "Pipeline")
    except (ImportError, AttributeError) as import_error:
        raise ImportError(
            "Pro modelovou diarizaci nainstaluj: pip install pyannote.audio"
        ) from import_error

    pipeline = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)
    if device:
        pipeline.to(torch.device(device))

    diarization = pipeline(str(audio_path))
    diarization_turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diarization_turns.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            }
        )

    return diarization_turns


def overlap_seconds(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def relabel_to_project_labels(segments):
    speaker_order = []
    mapping = {}

    for segment in segments:
        raw = segment.get("speaker", "unknown")
        if raw == "unknown":
            continue
        if raw not in speaker_order:
            speaker_order.append(raw)

    if len(speaker_order) >= 1:
        mapping[speaker_order[0]] = "interviewer"
    if len(speaker_order) >= 2:
        mapping[speaker_order[1]] = "interviewee"

    relabeled = []
    for segment in segments:
        current = segment.get("speaker", "unknown")
        project_label = mapping.get(current, "unknown")
        segment_copy = dict(segment)
        segment_copy["speaker"] = project_label
        segment_copy["speakers"] = [project_label]
        relabeled.append(segment_copy)

    return relabeled, mapping


def assign_speakers_by_overlap(asr_segments, diarization_turns, min_overlap=0.05):
    assigned = []

    for segment in asr_segments:
        if not isinstance(segment, dict):
            continue

        seg_start = float(segment.get("start", 0.0))
        seg_end = float(segment.get("end", seg_start))

        overlap_by_speaker = {}
        for diar in diarization_turns:
            diar_start = float(diar["start"])
            diar_end = float(diar["end"])
            ov = overlap_seconds(seg_start, seg_end, diar_start, diar_end)
            if ov <= 0.0:
                continue
            speaker = diar["speaker"]
            overlap_by_speaker[speaker] = overlap_by_speaker.get(speaker, 0.0) + ov

        chosen = "unknown"
        if overlap_by_speaker:
            candidate, value = max(overlap_by_speaker.items(), key=lambda item: item[1])
            if value >= min_overlap:
                chosen = candidate

        segment_copy = dict(segment)
        segment_copy["speaker"] = chosen
        segment_copy["speakers"] = [chosen]
        assigned.append(segment_copy)

    relabeled, mapping = relabel_to_project_labels(assigned)
    return relabeled, mapping


def compute_speaker_stats(segments):
    stats = {}
    for segment in segments:
        speaker = segment.get("speaker", "unknown")
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        duration = max(0.0, end - start)

        if speaker not in stats:
            stats[speaker] = {"segments": 0, "seconds": 0.0}

        stats[speaker]["segments"] += 1
        stats[speaker]["seconds"] += duration

    for speaker in stats:
        stats[speaker]["seconds"] = round(stats[speaker]["seconds"], 3)

    return stats


def build_output_payload(input_data, diarized_segments, stats, input_path, diarization_meta):
    metadata = dict(input_data.get("metadata", {})) if isinstance(input_data, dict) else {}
    metadata.update(
        {
            "diarization": diarization_meta,
            "source_json": str(input_path),
        }
    )

    full_transcription = " ".join(segment.get("text", "") for segment in diarized_segments).strip()

    return {
        "metadata": metadata,
        "segments": diarized_segments,
        "full_transcription": full_transcription,
        "speaker_stats": stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Experimentální mono diarizace nad ASR segmenty")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Vstupní ASR JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Výstupní diarized JSON")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO, help="MONO audio pro modelovou diarizaci")
    parser.add_argument("--method", choices=["pyannote", "heuristic"], default="pyannote", help="Metoda diarizace")
    parser.add_argument("--gap-switch-seconds", type=float, default=1.5, help="Pauza (s) pro heuristic přepnutí")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face token pro pyannote")
    parser.add_argument("--pyannote-model", type=str, default=DEFAULT_PYANNOTE_MODEL, help="Pyannote model ID")
    parser.add_argument("--device", type=str, default="cpu", help="Zařízení pro pyannote (cpu/cuda)")
    parser.add_argument("--min-overlap", type=float, default=0.05, help="Min. překryv (s) pro přiřazení speakeru")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Vstupní soubor neexistuje: {args.input}")

    input_data = load_json(args.input)
    segments = input_data.get("segments", []) if isinstance(input_data, dict) else []

    if args.method == "pyannote":
        hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        diar_turns = run_pyannote(
            audio_path=args.audio,
            model_name=args.pyannote_model,
            hf_token=hf_token,
            device=args.device,
        )
        diarized_segments, mapping = assign_speakers_by_overlap(
            segments,
            diar_turns,
            min_overlap=args.min_overlap,
        )
        diarization_meta = {
            "method": "pyannote-overlap-assignment",
            "pyannote_model": args.pyannote_model,
            "device": args.device,
            "min_overlap_seconds": args.min_overlap,
            "raw_speaker_mapping": mapping,
            "turn_count": len(diar_turns),
            "is_experimental": True,
        }
    else:
        diarized_segments = diarize_heuristic(segments, gap_switch_seconds=args.gap_switch_seconds)
        diarization_meta = {
            "method": "heuristic-gap-and-question-switch",
            "gap_switch_seconds": args.gap_switch_seconds,
            "labels": ["interviewer", "interviewee"],
            "is_experimental": True,
        }

    stats = compute_speaker_stats(diarized_segments)
    output_payload = build_output_payload(
        input_data,
        diarized_segments,
        stats,
        input_path=args.input,
        diarization_meta=diarization_meta,
    )
    save_json(args.output, output_payload)

    print("Hotovo: experimentální mono diarizace")
    print(f"Metoda: {args.method}")
    print(f"Vstup:  {args.input}")
    print(f"Výstup: {args.output}")
    print(f"Segmentů: {len(diarized_segments)}")
    print("Speaker stats:")
    for speaker, speaker_stats in stats.items():
        print(f"- {speaker}: segments={speaker_stats['segments']}, seconds={speaker_stats['seconds']}")


if __name__ == "__main__":
    main()
