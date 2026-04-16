import json
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from jiwer import process_words

WORD_CHANGES_PATH = Path("data/WordChanges.txt")
WORD_CHANGES_CACHE = None

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

NUMBER_WORDS = {
    "nula", "jeden", "jedna", "jedno", "dva", "dvě", "tri", "tři", "čtyři", "ctyri", "pet", "pět",
    "šest", "sest", "sedm", "osm", "devět", "devet", "deset", "jedenáct", "jedenact", "dvanáct", "dvanact",
    "třináct", "trinact", "čtrnáct", "ctrnact", "patnáct", "patnact", "šestnáct", "sestnact", "sedmnáct", "sedmnact",
    "osmnáct", "osmnact", "devatenáct", "devatenact", "devatenácet", "dvacet", "třicet", "tricet", "čtyřicet",
    "ctyricet", "padesát", "padesat", "šedesát", "sedesat", "sedmdesát", "sedmdesat", "osmdesát", "osmdesat",
    "devadesát", "devadesat", "stě", "ste", "sto", "set", "tisíc", "tisic",
}

NUMBER_STEMS_ASCII = {
    "nul", "jedn", "dva", "dv", "tri", "ctyr", "pet", "sest", "sedm", "osm", "devet",
    "deset", "jedenact", "dvanact", "trinact", "ctrnact", "patnact", "sestnact", "sedmnact",
    "osmnact", "devatenact", "devatenacet", "dvacet", "tricet", "ctyricet", "padesat", "sedesat",
    "sedmdesat", "osmdesat", "devadesat", "sto", "ste", "set", "tisic",
}

NUMBER_SUFFIXES_ASCII = (
    "eho", "emu", "em", "ym", "ych", "ymi", "y", "a", "u", "ou", "i", "o", "ho",
    "teho", "ateho", "cateho", "nacteho",
)


def clean_text(text):
    return str(text or "").strip()


def strip_diacritics(text):
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


NUMBER_WORDS_ASCII = {strip_diacritics(word) for word in NUMBER_WORDS}


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
            if not line:
                continue

            parts = re.split(r"\t+", line)
            if len(parts) < 2:
                continue

            canonical = parts[0].strip().lower()
            variant = parts[1].strip().lower()
            if canonical and variant:
                mapping[variant] = canonical

    WORD_CHANGES_CACHE = mapping
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


def is_numeric_like_token(token):
    if not token:
        return False
    if token.isdigit():
        return True
    if re.fullmatch(r"\d+[a-z]{0,3}", token):
        return True

    token_ascii = strip_diacritics(token)
    if token_ascii in NUMBER_WORDS_ASCII:
        return True

    for suffix in NUMBER_SUFFIXES_ASCII:
        if token_ascii.endswith(suffix):
            stem = token_ascii[:-len(suffix)]
            if len(stem) >= 3 and stem in NUMBER_STEMS_ASCII:
                return True

    return False


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


def normalize_numbers_to_placeholder(tokens):
    normalized = ["<num>" if is_numeric_like_token(token) else token for token in tokens]
    compacted = []
    for token in normalized:
        if token == "<num>" and compacted and compacted[-1] == "<num>":
            continue
        compacted.append(token)
    return compacted


def normalize_text(text):
    tokens = normalize_common(text).split()
    tokens = collapse_letter_spelling(tokens)
    tokens = normalize_numbers_to_placeholder(tokens)
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


def evaluate_mapping(asr_speaker_texts, gt_speaker_texts, mapping, word_changes_map):
    per_role_result = {}
    for role in ("interviewer", "interviewee"):
        asr_key = mapping[role]
        ref_text = gt_speaker_texts[role]
        hyp_text = asr_speaker_texts[asr_key]
        per_role_result[role] = evaluate_pair(ref_text, hyp_text, word_changes_map)

    summary = compute_macro_and_weighted_wer(per_role_result)
    return per_role_result, summary


def evaluate_pair(reference_text, hypothesis_text, word_changes_map):
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

    if word_changes_map:
        corrected_hyp, runtime_info = apply_word_changes_from_substitutions(norm_ref, norm_hyp, word_changes_map)
        word_changes_runtime.update(runtime_info)
        word_changes_runtime["wer_before"] = base_diagnostics["wer"]

        if corrected_hyp and corrected_hyp != norm_hyp:
            corrected_diagnostics = build_error_diagnostics(norm_ref, corrected_hyp)
            if corrected_diagnostics["wer"] <= base_diagnostics["wer"]:
                norm_hyp = corrected_hyp
                final_diagnostics = corrected_diagnostics

        word_changes_runtime["wer_after"] = final_diagnostics["wer"]

    ref_words = norm_ref.split()
    hyp_words = norm_hyp.split()

    return {
        "diagnostics": final_diagnostics,
        "word_changes_runtime": word_changes_runtime,
        "reference_word_count": len(ref_words),
        "hypothesis_word_count": len(hyp_words),
        "reference_head": " ".join(ref_words[:15]) + "...",
        "hypothesis_head": " ".join(hyp_words[:15]) + "...",
        "reference_tail": "..." + " ".join(ref_words[-15:]),
        "hypothesis_tail": "..." + " ".join(hyp_words[-15:]),
        "norm_ref": norm_ref,
        "norm_hyp": norm_hyp,
    }


def build_speaker_report_section(label, role, asr_key, pair_result, word_changes_entries):
    diag = pair_result["diagnostics"]
    runtime = pair_result["word_changes_runtime"]

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
    metadata,
    recording_id,
    mapping,
    per_role_result,
    summary,
    mapping_confidence,
    word_changes_entries,
):
    lines = []
    lines.append("ASR Speaker Evaluation Report")
    lines.append("=" * 80)
    lines.append(f"Source JSON: {hyp_file}")
    lines.append("Reference (GT): results/ground_truth_speakers.json")
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
        )
    )

    with open(path, "w", encoding="utf-8") as file_handle:
        file_handle.write("\n".join(lines) + "\n")


def evaluate_speakers(hyp_file):
    if not os.path.exists(hyp_file):
        print(f"Chyba: Soubor {hyp_file} nebyl nalezen!")
        return

    gt_file = "results/ground_truth_speakers.json"
    if not os.path.exists(gt_file):
        print(f"Chyba: Referenční soubor nebyl nalezen: {gt_file}")
        return

    hyp_data = load_json(hyp_file)
    gt_data = load_json(gt_file)

    recording_id = extract_recording_id_from_filename(hyp_file)

    asr_speaker_texts = collect_asr_full_texts(hyp_data)
    gt_speaker_texts = load_gt_speaker_texts(gt_data, recording_id=recording_id)

    mapping = assign_roles_by_length(asr_speaker_texts, gt_speaker_texts)
    swapped = swapped_mapping(mapping)

    word_changes_map = load_word_changes_map()
    word_changes_entries = len(word_changes_map)

    per_role_result, summary = evaluate_mapping(
        asr_speaker_texts=asr_speaker_texts,
        gt_speaker_texts=gt_speaker_texts,
        mapping=mapping,
        word_changes_map=word_changes_map,
    )
    _, swapped_summary = evaluate_mapping(
        asr_speaker_texts=asr_speaker_texts,
        gt_speaker_texts=gt_speaker_texts,
        mapping=swapped,
        word_changes_map=word_changes_map,
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

    report_path = diagnostics_output_path(hyp_file)
    metadata = hyp_data.get("metadata", {}) if isinstance(hyp_data, dict) else {}

    write_report(
        path=report_path,
        hyp_file=hyp_file,
        metadata=metadata,
        recording_id=recording_id,
        mapping=mapping,
        per_role_result=per_role_result,
        summary=summary,
        mapping_confidence=mapping_confidence,
        word_changes_entries=word_changes_entries,
    )

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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Použití: python scripts/evaluate_wer_speakers.py results/vysledek_individual_full.json")
    else:
        evaluate_speakers(sys.argv[1])
