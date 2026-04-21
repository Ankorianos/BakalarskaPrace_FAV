import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

from jiwer import process_words

WORD_CHANGES_PATH = Path("data/WordChanges.txt")
WORD_CHANGES_CACHE = None
NUMBER_CHANGES_PATH = Path("data/NumberChanges.txt")
NUMBER_CHANGES_CACHE = None

HALLUCINATIONS = [
    r"titulky vytvořil.*",
    r"děkuji za sledování.*",
    r"odběratelé.*",
    r"přeložil.*",
    r"reaping.*",
    r"watch next.*",
    r"thanks for watching.*",
    r"titulky.*",
]

LETTER_NAME_MAP = {
    "a": "a", "á": "a", "bé": "b", "b": "b", "cé": "c", "c": "c", "čé": "č", "č": "č",
    "dé": "d", "d": "d", "é": "e", "e": "e", "ef": "f", "f": "f", "gé": "g", "g": "g",
    "há": "h", "h": "h", "chá": "ch", "í": "i", "i": "i", "jé": "j", "j": "j", "ká": "k", "k": "k",
    "el": "l", "l": "l", "em": "m", "m": "m", "en": "n", "n": "n", "ó": "o", "o": "o",
    "pé": "p", "p": "p", "kvé": "q", "q": "q", "er": "r", "r": "r", "es": "s", "s": "s",
    "té": "t", "t": "t", "ú": "u", "u": "u", "ů": "u", "vé": "v", "v": "v", "w": "w", "dvojitévé": "w",
    "ix": "x", "x": "x", "ý": "y", "y": "y", "zet": "z", "z": "z", "žet": "ž", "ž": "ž",
}

def clean_text(text):
    return str(text or "").strip()


def load_json(path):
    with open(path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def extract_recording_id_from_filename(path_like):
    stem = Path(path_like).stem
    match = re.match(r"^(\d+_\d+)", stem)
    if not match:
        match = re.match(r"^(\d+)", stem)
    return match.group(1) if match else None


def diagnostics_output_path(hyp_file):
    stem = Path(hyp_file).stem
    safe_stem = re.sub(r"[^\w.-]+", "_", stem).strip("._") or "hypothesis"
    return f"results/eval_report_speakers_{safe_stem}.txt"


def resolve_speaker_gt_file(recording_id):
    if not recording_id:
        return None

    path = f"results/{recording_id}_gt_speakers.json"
    return path if os.path.exists(path) else None


def resolve_speaker_gt_files(recording_id):
    files = []

    primary = resolve_speaker_gt_file(recording_id)
    if primary:
        files.append(primary)

    formal_all = "results/all_formal_gt_speakers.json"
    if os.path.exists(formal_all):
        files.append(formal_all)

    return files


def is_whisper_backend(metadata):
    if not isinstance(metadata, dict):
        return False

    backend_candidates = [
        metadata.get("backend"),
        metadata.get("model"),
        metadata.get("asr_backend"),
    ]
    haystack = " ".join(str(item or "") for item in backend_candidates).lower()
    return "whisper" in haystack


def load_word_changes_map():
    global WORD_CHANGES_CACHE
    if WORD_CHANGES_CACHE is not None:
        return WORD_CHANGES_CACHE

    mapping = {}
    if not WORD_CHANGES_PATH.exists():
        WORD_CHANGES_CACHE = mapping
        return mapping

    with open(WORD_CHANGES_PATH, "r", encoding="utf-8", errors="ignore") as file_handle:
        for source_line in file_handle:
            line = source_line.strip()
            if not line or line.startswith("#"):
                continue

            canonical = ""
            variant = ""

            if "\t" in line:
                parts = re.split(r"\t+", line)
                if len(parts) >= 2:
                    canonical = parts[0].strip().lower()
                    variant = parts[1].strip().lower()
            else:
                parts = line.split()
                if len(parts) == 2:
                    canonical = parts[0].strip().lower()
                    variant = parts[1].strip().lower()

            if canonical and variant:
                mapping[variant] = canonical

    WORD_CHANGES_CACHE = mapping
    return mapping


def load_number_changes_map():
    global NUMBER_CHANGES_CACHE
    if NUMBER_CHANGES_CACHE is not None:
        return NUMBER_CHANGES_CACHE

    mapping = {}
    if not NUMBER_CHANGES_PATH.exists():
        NUMBER_CHANGES_CACHE = mapping
        return mapping

    with open(NUMBER_CHANGES_PATH, "r", encoding="utf-8", errors="ignore") as file_handle:
        for source_line in file_handle:
            line = source_line.strip()
            if not line or line.startswith("#"):
                continue

            canonical = ""
            variant = ""

            if "\t" in line:
                parts = re.split(r"\t+", line)
                if len(parts) >= 2:
                    canonical = parts[0].strip().lower()
                    variant = parts[1].strip().lower()
            else:
                parts = line.split()
                if len(parts) == 2:
                    canonical = parts[0].strip().lower()
                    variant = parts[1].strip().lower()

            if canonical and variant:
                mapping[variant] = canonical

    NUMBER_CHANGES_CACHE = mapping
    return mapping


def canonicalize_word_change_token(token, mapping):
    normalized = clean_text(token).lower()
    return mapping.get(normalized, normalized)


def are_word_change_equivalent(left_token, right_token, mapping):
    if not left_token or not right_token or not mapping:
        return False
    return canonicalize_word_change_token(left_token, mapping) == canonicalize_word_change_token(right_token, mapping)


def apply_word_changes_from_substitutions(reference_text, hypothesis_text, mapping):
    if not reference_text or not hypothesis_text or not mapping:
        return hypothesis_text, {"substitute_spans_checked": 0, "tokens_replaced": 0}

    output = process_words(reference_text, hypothesis_text)
    ref_words = reference_text.split()
    hyp_words = hypothesis_text.split()
    corrected_hyp_words = list(hyp_words)

    substitute_spans_checked = 0
    tokens_replaced = 0

    for alignment_group in output.alignments:
        for change in alignment_group:
            if change.type != "substitute":
                continue

            substitute_spans_checked += 1
            ref_span = ref_words[change.ref_start_idx:change.ref_end_idx]
            hyp_span = hyp_words[change.hyp_start_idx:change.hyp_end_idx]
            if len(ref_span) != len(hyp_span):
                continue

            for idx, (ref_token, hyp_token) in enumerate(zip(ref_span, hyp_span)):
                if not are_word_change_equivalent(ref_token, hyp_token, mapping):
                    continue
                target_idx = change.hyp_start_idx + idx
                if 0 <= target_idx < len(corrected_hyp_words) and corrected_hyp_words[target_idx] != ref_token:
                    corrected_hyp_words[target_idx] = ref_token
                    tokens_replaced += 1

    return " ".join(corrected_hyp_words).strip(), {
        "substitute_spans_checked": substitute_spans_checked,
        "tokens_replaced": tokens_replaced,
    }


def normalize_common(text):
    normalized = clean_text(text).lower()
    for pattern in HALLUCINATIONS:
        normalized = re.sub(pattern, " ", normalized, flags=re.IGNORECASE)
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def collapse_letter_spelling(tokens):
    mapped = [LETTER_NAME_MAP.get(token, token) for token in tokens]
    output = []
    index = 0

    while index < len(mapped):
        run = []
        while index < len(mapped) and len(mapped[index]) == 1 and mapped[index].isalpha():
            run.append(mapped[index])
            index += 1

        if len(run) >= 2:
            output.append("".join(run))
            continue
        if len(run) == 1:
            output.append(run[0])
            continue

        output.append(mapped[index])
        index += 1

    return output


def normalize_text(text):
    tokens = normalize_common(text).split()
    tokens = collapse_letter_spelling(tokens)
    return " ".join(tokens).strip()


def build_error_diagnostics(reference_text, hypothesis_text):
    output = process_words(reference_text, hypothesis_text)
    ref_words = reference_text.split()
    hyp_words = hypothesis_text.split()

    substitutions = Counter()
    insertions = Counter()
    deletions = Counter()

    for alignment_group in output.alignments:
        for change in alignment_group:
            if change.type == "substitute":
                left = " ".join(ref_words[change.ref_start_idx:change.ref_end_idx])
                right = " ".join(hyp_words[change.hyp_start_idx:change.hyp_end_idx])
                substitutions[(left, right)] += 1
            elif change.type == "insert":
                token = " ".join(hyp_words[change.hyp_start_idx:change.hyp_end_idx])
                if token:
                    insertions[token] += 1
            elif change.type == "delete":
                token = " ".join(ref_words[change.ref_start_idx:change.ref_end_idx])
                if token:
                    deletions[token] += 1

    return {
        "wer": output.wer,
        "hits": output.hits,
        "substitutions": output.substitutions,
        "insertions": output.insertions,
        "deletions": output.deletions,
        "top_substitutions": substitutions.most_common(20),
        "top_insertions": insertions.most_common(20),
        "top_deletions": deletions.most_common(20),
    }


def build_qualitative_examples(diagnostics, max_examples=3):
    examples = []

    for (left, right), count in diagnostics.get("top_substitutions", []):
        if left and right:
            examples.append(f"SUB ({count}x): '{left}' -> '{right}'")
            break

    for token, count in diagnostics.get("top_deletions", []):
        if token:
            examples.append(f"DEL ({count}x): '{token}'")
            break

    for token, count in diagnostics.get("top_insertions", []):
        if token:
            examples.append(f"INS ({count}x): '{token}'")
            break

    return examples[:max_examples]


def collect_asr_full_texts(hyp_data):
    result = {}

    if isinstance(hyp_data, dict):
        for key, value in hyp_data.items():
            if not isinstance(value, str):
                continue
            if re.fullmatch(r"(?i)speaker_[lr]_full_transcription", key):
                normalized_key = key.lower()
                result[normalized_key] = clean_text(value)

    if result:
        return result

    segments = hyp_data.get("segments", []) if isinstance(hyp_data, dict) else []
    buckets = {}
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        speaker = clean_text(segment.get("speaker", ""))
        text = clean_text(segment.get("text", ""))
        if not speaker or not text:
            continue
        buckets.setdefault(speaker, []).append(text)

    for speaker, parts in buckets.items():
        result[f"speaker_{speaker.lower()}_full_transcription"] = " ".join(parts).strip()

    return result


def load_gt_speaker_texts(gt_speakers_data, recording_id=None):
    if not isinstance(gt_speakers_data, dict):
        return {}

    selected = None
    if recording_id and recording_id in gt_speakers_data and isinstance(gt_speakers_data[recording_id], dict):
        selected = gt_speakers_data[recording_id]
    else:
        for value in gt_speakers_data.values():
            if isinstance(value, dict) and "interviewer" in value and "interviewee" in value:
                selected = value
                break

    if not isinstance(selected, dict):
        return {}

    result = {}
    for role in ("interviewer", "interviewee"):
        node = selected.get(role)
        if isinstance(node, dict):
            result[role] = clean_text(node.get("text", ""))

    return result


def assign_roles_by_length(asr_texts, gt_texts):
    asr_items = [(key, text) for key, text in asr_texts.items() if text]
    if len(asr_items) < 2:
        raise ValueError("V ASR JSON nebyly nalezeny dva full přepisy mluvčích.")

    if not gt_texts.get("interviewer") or not gt_texts.get("interviewee"):
        raise ValueError("V ground_truth_speakers.json chybí interviewer/interviewee text.")

    asr_sorted = sorted(asr_items, key=lambda item: len(item[1]))
    gt_sorted = sorted(((role, text) for role, text in gt_texts.items()), key=lambda item: len(item[1]))

    mapping = {
        gt_sorted[0][0]: asr_sorted[0][0],
        gt_sorted[-1][0]: asr_sorted[-1][0],
    }

    return mapping


def swapped_mapping(mapping):
    return {
        "interviewer": mapping["interviewee"],
        "interviewee": mapping["interviewer"],
    }


def compute_macro_and_weighted_wer(per_role_result):
    interviewer_wer = per_role_result["interviewer"]["diagnostics"]["wer"]
    interviewee_wer = per_role_result["interviewee"]["diagnostics"]["wer"]
    macro_wer = (interviewer_wer + interviewee_wer) / 2.0

    total_ref_words = (
        per_role_result["interviewer"]["reference_word_count"]
        + per_role_result["interviewee"]["reference_word_count"]
    )
    weighted_wer = 0.0
    if total_ref_words > 0:
        weighted_wer = (
            interviewer_wer * per_role_result["interviewer"]["reference_word_count"]
            + interviewee_wer * per_role_result["interviewee"]["reference_word_count"]
        ) / total_ref_words

    return {
        "interviewer_wer": interviewer_wer,
        "interviewee_wer": interviewee_wer,
        "macro_wer": macro_wer,
        "weighted_wer": weighted_wer,
        "total_ref_words": total_ref_words,
    }


def evaluate_mapping(asr_speaker_texts, gt_speaker_texts, mapping, word_changes_map, number_changes_map):
    per_role_result = {}
    for role in ("interviewer", "interviewee"):
        asr_key = mapping[role]
        ref_text = gt_speaker_texts[role]
        hyp_text = asr_speaker_texts[asr_key]
        per_role_result[role] = evaluate_pair(ref_text, hyp_text, word_changes_map, number_changes_map)

    summary = compute_macro_and_weighted_wer(per_role_result)
    return per_role_result, summary


def evaluate_pair(reference_text, hypothesis_text, word_changes_map, number_changes_map):
    norm_ref = normalize_text(reference_text)
    norm_hyp = normalize_text(hypothesis_text)

    base_diagnostics = build_error_diagnostics(norm_ref, norm_hyp)
    final_diagnostics = base_diagnostics

    word_changes_runtime = {
        "substitute_spans_checked": 0,
        "tokens_replaced": 0,
        "wer_before": None,
        "wer_after": None,
    }

    number_changes_runtime = {
        "substitute_spans_checked": 0,
        "tokens_replaced": 0,
        "wer_before": None,
        "wer_after": None,
    }

    if number_changes_map:
        corrected_hyp, runtime_info = apply_word_changes_from_substitutions(norm_ref, norm_hyp, number_changes_map)
        number_changes_runtime.update(runtime_info)
        number_changes_runtime["wer_before"] = final_diagnostics["wer"]

        if corrected_hyp and corrected_hyp != norm_hyp:
            corrected_diagnostics = build_error_diagnostics(norm_ref, corrected_hyp)
            if corrected_diagnostics["wer"] <= final_diagnostics["wer"]:
                norm_hyp = corrected_hyp
                final_diagnostics = corrected_diagnostics

        number_changes_runtime["wer_after"] = final_diagnostics["wer"]

    if word_changes_map:
        corrected_hyp, runtime_info = apply_word_changes_from_substitutions(norm_ref, norm_hyp, word_changes_map)
        word_changes_runtime.update(runtime_info)
        word_changes_runtime["wer_before"] = final_diagnostics["wer"]

        if corrected_hyp and corrected_hyp != norm_hyp:
            corrected_diagnostics = build_error_diagnostics(norm_ref, corrected_hyp)
            if corrected_diagnostics["wer"] <= final_diagnostics["wer"]:
                norm_hyp = corrected_hyp
                final_diagnostics = corrected_diagnostics

        word_changes_runtime["wer_after"] = final_diagnostics["wer"]

    ref_words = norm_ref.split()
    hyp_words = norm_hyp.split()

    return {
        "diagnostics": final_diagnostics,
        "word_changes_runtime": word_changes_runtime,
        "number_changes_runtime": number_changes_runtime,
        "reference_word_count": len(ref_words),
        "hypothesis_word_count": len(hyp_words),
        "reference_head": " ".join(ref_words[:15]) + "...",
        "hypothesis_head": " ".join(hyp_words[:15]) + "...",
        "reference_tail": "..." + " ".join(ref_words[-15:]),
        "hypothesis_tail": "..." + " ".join(hyp_words[-15:]),
        "norm_ref": norm_ref,
        "norm_hyp": norm_hyp,
    }


def build_speaker_report_section(label, role, asr_key, pair_result, word_changes_entries, number_changes_entries):
    diag = pair_result["diagnostics"]
    runtime = pair_result["word_changes_runtime"]
    number_runtime = pair_result["number_changes_runtime"]

    lines = []
    lines.append(f"{label}")
    lines.append("-" * 80)
    lines.append(f"GT role: {role} | ASR source: {asr_key}")
    lines.append(f"WER: {diag['wer'] * 100:.2f} %")
    lines.append(f"Word counts: REF={pair_result['reference_word_count']} | HYP={pair_result['hypothesis_word_count']}")
    lines.append(
        "WordChanges: enabled={enabled} | entries={entries} | checked_sub_spans={checked} | replaced_tokens={replaced}".format(
            enabled=bool(word_changes_entries),
            entries=word_changes_entries,
            checked=runtime.get("substitute_spans_checked", 0),
            replaced=runtime.get("tokens_replaced", 0),
        )
    )

    lines.append(
        "NumberChanges: enabled={enabled} | entries={entries} | checked_sub_spans={checked} | replaced_tokens={replaced}".format(
            enabled=bool(number_changes_entries),
            entries=number_changes_entries,
            checked=number_runtime.get("substitute_spans_checked", 0),
            replaced=number_runtime.get("tokens_replaced", 0),
        )
    )

    number_before = number_runtime.get("wer_before")
    number_after = number_runtime.get("wer_after")
    if number_before is not None and number_after is not None:
        lines.append(
            "NumberChanges impact (WER): before={before:.2f} % | after={after:.2f} % | delta={delta:+.2f} p.b.".format(
                before=number_before * 100,
                after=number_after * 100,
                delta=(number_after - number_before) * 100,
            )
        )

    before = runtime.get("wer_before")
    after = runtime.get("wer_after")
    if before is not None and after is not None:
        lines.append(
            "WordChanges impact (WER): before={before:.2f} % | after={after:.2f} % | delta={delta:+.2f} p.b.".format(
                before=before * 100,
                after=after * 100,
                delta=(after - before) * 100,
            )
        )

    lines.append(
        f"HITS: {diag['hits']} | SUB: {diag['substitutions']} | INS: {diag['insertions']} | DEL: {diag['deletions']}"
    )

    lines.append("")
    lines.append("Kvalitativní ukázky (2-3):")
    examples = build_qualitative_examples(diag, max_examples=3)
    if examples:
        for idx, item in enumerate(examples, start=1):
            lines.append(f"{idx}. {item}")
    else:
        lines.append("(Nenalezeny žádné ukázky)")

    lines.append("")
    lines.append("Začátek srovnání:")
    lines.append(f"REF: {pair_result['reference_head']}")
    lines.append(f"HYP: {pair_result['hypothesis_head']}")

    lines.append("")
    lines.append("Konec srovnání:")
    lines.append(f"REF: {pair_result['reference_tail']}")
    lines.append(f"HYP: {pair_result['hypothesis_tail']}")

    lines.append("")
    lines.append("Top substitutions:")
    for (left, right), count in diag["top_substitutions"]:
        lines.append(f"{count}x | {left} => {right}")

    lines.append("")
    lines.append("Top deletions:")
    for token, count in diag["top_deletions"]:
        lines.append(f"{count}x | {token}")

    lines.append("")
    lines.append("Top insertions:")
    for token, count in diag["top_insertions"]:
        lines.append(f"{count}x | {token}")

    return lines


def write_report(
    path,
    hyp_file,
    gt_file,
    metadata,
    recording_id,
    mapping,
    per_role_result,
    summary,
    mapping_confidence,
    word_changes_entries,
    number_changes_entries,
    append=False,
    section_title=None,
):
    lines = []
    lines.append(section_title or "ASR Speaker Evaluation Report")
    lines.append("=" * 80)
    lines.append(f"Source JSON: {hyp_file}")
    lines.append(f"Reference (GT): {gt_file}")
    lines.append(f"Recording ID: {recording_id or 'unknown'}")
    lines.append("Mode: {mode} | Model: {model} | Backend: {backend}".format(
        mode=metadata.get("mode", "unknown") if isinstance(metadata, dict) else "unknown",
        model=metadata.get("model", "unknown") if isinstance(metadata, dict) else "unknown",
        backend=metadata.get("backend", "unknown") if isinstance(metadata, dict) else "unknown",
    ))
    lines.append("Runtime: {runtime}s | Run started: {started} | Run finished: {finished}".format(
        runtime=metadata.get("runtime_seconds", "n/a") if isinstance(metadata, dict) else "n/a",
        started=metadata.get("run_started_utc", "n/a") if isinstance(metadata, dict) else "n/a",
        finished=metadata.get("run_finished_utc", "n/a") if isinstance(metadata, dict) else "n/a",
    ))
    lines.append("Evaluation source: speaker_full_transcriptions")
    lines.append("Speaker mapping by text length:")
    lines.append(f"- interviewer <= {mapping['interviewer']}")
    lines.append(f"- interviewee <= {mapping['interviewee']}")
    lines.append(
        "Mapping confidence (weighted WER): length_based={lb:.2f} % | swapped={sw:.2f} % | winner={winner} | delta={delta:.2f} p.b.".format(
            lb=mapping_confidence["length_based_weighted_wer"] * 100,
            sw=mapping_confidence["swapped_weighted_wer"] * 100,
            winner=mapping_confidence["winner"],
            delta=mapping_confidence["delta_percentage_points"],
        )
    )
    lines.append("-" * 80)

    lines.append(f"WER (interviewer): {summary['interviewer_wer'] * 100:.2f} %")
    lines.append(f"WER (interviewee): {summary['interviewee_wer'] * 100:.2f} %")
    lines.append(f"WER (macro avg): {summary['macro_wer'] * 100:.2f} %")
    lines.append(f"WER (weighted by REF words): {summary['weighted_wer'] * 100:.2f} %")

    lines.append("")
    lines.append("=" * 80)
    lines.append("")

    lines.extend(
        build_speaker_report_section(
            label="Speaker section: interviewer",
            role="interviewer",
            asr_key=mapping["interviewer"],
            pair_result=per_role_result["interviewer"],
            word_changes_entries=word_changes_entries,
            number_changes_entries=number_changes_entries,
        )
    )

    lines.append("")
    lines.append("=" * 80)
    lines.append("")

    lines.extend(
        build_speaker_report_section(
            label="Speaker section: interviewee",
            role="interviewee",
            asr_key=mapping["interviewee"],
            pair_result=per_role_result["interviewee"],
            word_changes_entries=word_changes_entries,
            number_changes_entries=number_changes_entries,
        )
    )

    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as file_handle:
        if append:
            divider = ("=" * 80 + "\n") * 5
            file_handle.write("\n" + divider + "\n")
        file_handle.write("\n".join(lines) + "\n")


def evaluate_speakers(hyp_file):
    if not os.path.exists(hyp_file):
        print(f"Chyba: Soubor {hyp_file} nebyl nalezen!")
        return

    recording_id = extract_recording_id_from_filename(hyp_file)
    gt_files = resolve_speaker_gt_files(recording_id)
    if not gt_files:
        print("Chyba: Referenční speaker GT soubor nebyl nalezen.")
        print("Očekáváno: results/<name_part>_gt_speakers.json a/nebo results/all_formal_gt_speakers.json")
        return

    hyp_data = load_json(hyp_file)
    report_path = diagnostics_output_path(hyp_file)
    metadata = hyp_data.get("metadata", {}) if isinstance(hyp_data, dict) else {}

    asr_speaker_texts = collect_asr_full_texts(hyp_data)
    word_changes_map = load_word_changes_map()
    word_changes_entries = len(word_changes_map)
    number_changes_map = load_number_changes_map() if is_whisper_backend(metadata) else {}
    number_changes_entries = len(number_changes_map)

    wrote_any = False
    for gt_file in gt_files:
        gt_data = load_json(gt_file)
        gt_speaker_texts = load_gt_speaker_texts(gt_data, recording_id=recording_id)
        if not gt_speaker_texts.get("interviewer") or not gt_speaker_texts.get("interviewee"):
            print(f"Přeskakuji {gt_file}: chybí interviewer/interviewee text pro recording {recording_id}.")
            continue

        mapping = assign_roles_by_length(asr_speaker_texts, gt_speaker_texts)
        swapped = swapped_mapping(mapping)

        per_role_result, summary = evaluate_mapping(
            asr_speaker_texts=asr_speaker_texts,
            gt_speaker_texts=gt_speaker_texts,
            mapping=mapping,
            word_changes_map=word_changes_map,
            number_changes_map=number_changes_map,
        )
        _, swapped_summary = evaluate_mapping(
            asr_speaker_texts=asr_speaker_texts,
            gt_speaker_texts=gt_speaker_texts,
            mapping=swapped,
            word_changes_map=word_changes_map,
            number_changes_map=number_changes_map,
        )

        length_based_weighted = summary["weighted_wer"]
        swapped_weighted = swapped_summary["weighted_wer"]
        winner = "length_based" if length_based_weighted <= swapped_weighted else "swapped"
        mapping_confidence = {
            "length_based_weighted_wer": length_based_weighted,
            "swapped_weighted_wer": swapped_weighted,
            "winner": winner,
            "delta_percentage_points": abs(length_based_weighted - swapped_weighted) * 100,
        }

        write_report(
            path=report_path,
            hyp_file=hyp_file,
            gt_file=gt_file,
            metadata=metadata,
            recording_id=recording_id,
            mapping=mapping,
            per_role_result=per_role_result,
            summary=summary,
            mapping_confidence=mapping_confidence,
            word_changes_entries=word_changes_entries,
            number_changes_entries=number_changes_entries,
            append=wrote_any,
            section_title=f"ASR Speaker Evaluation Report | {Path(gt_file).name}",
        )
        wrote_any = True

        print("=" * 80)
        print("SPEAKER EVALUATION")
        print(f"Hypotéza: {hyp_file}")
        print(f"Reference: {gt_file}")
        print(f"Mapování: interviewer <= {mapping['interviewer']} | interviewee <= {mapping['interviewee']}")
        print(f"WER interviewer: {summary['interviewer_wer'] * 100:.2f} %")
        print(f"WER interviewee: {summary['interviewee_wer'] * 100:.2f} %")
        print(
            "Mapping confidence (weighted WER): length_based={lb:.2f} % | swapped={sw:.2f} % | winner={winner} | delta={delta:.2f} p.b.".format(
                lb=mapping_confidence["length_based_weighted_wer"] * 100,
                sw=mapping_confidence["swapped_weighted_wer"] * 100,
                winner=mapping_confidence["winner"],
                delta=mapping_confidence["delta_percentage_points"],
            )
        )
        print(f"Report uložen do: {report_path}")
        print("=" * 80)

    if not wrote_any:
        print("Chyba: Nepodařilo se sestavit speaker evaluaci pro žádný GT soubor.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Použití: python scripts/evaluate_wer_speakers.py results/vysledek_individual_full.json")
    else:
        evaluate_speakers(sys.argv[1])
