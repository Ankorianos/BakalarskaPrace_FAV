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

MAX_WINDOW_OVERFLOW_SECONDS = 0.6
OFFSET_MAX_TRIM_WORDS = 6
OFFSET_MIN_IMPROVEMENT = 0.01
OFFSET_PREFIX_WINDOW = 12


def strip_diacritics(text):
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


NUMBER_WORDS_ASCII = {strip_diacritics(word) for word in NUMBER_WORDS}


def clean_text(text):
    return str(text or "").strip()


def load_json(path):
    with open(path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def diagnostics_output_path(hyp_file):
    stem = Path(hyp_file).stem
    safe_stem = re.sub(r"[^\w.-]+", "_", stem).strip("._") or "hypothesis"
    return f"results/eval_report_{safe_stem}.txt"


def extract_recording_id_from_filename(path_like):
    stem = Path(path_like).stem
    match = re.match(r"^(\d+_\d+)", stem)
    if not match:
        match = re.match(r"^(\d+)", stem)
    return match.group(1) if match else None


def is_full_hypothesis_file(path_like):
    stem = Path(path_like).stem.lower()
    return "_full_" in stem or stem.endswith("_full")


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
        if not token_ascii.endswith(suffix):
            continue
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


def prefix_match_count(left_tokens, right_tokens, window=OFFSET_PREFIX_WINDOW):
    compare_len = min(window, len(left_tokens), len(right_tokens))
    return sum(1 for idx in range(compare_len) if left_tokens[idx] == right_tokens[idx])


def apply_prefix_trim(text, trim_count):
    tokens = clean_text(text).split()
    return " ".join(tokens[max(trim_count, 0):]).strip()


def find_best_offset_compensation(norm_ref, norm_hyp):
    ref_tokens = norm_ref.split()
    hyp_tokens = norm_hyp.split()

    baseline = build_error_diagnostics(norm_ref, norm_hyp)
    base_wer = baseline["wer"]
    base_prefix_match = prefix_match_count(ref_tokens, hyp_tokens)

    best = {
        "ref_trim": 0,
        "hyp_trim": 0,
        "wer": base_wer,
        "prefix_match": base_prefix_match,
    }

    for trim in range(1, OFFSET_MAX_TRIM_WORDS + 1):
        for ref_trim, hyp_trim in ((trim, 0), (0, trim)):
            candidate_ref = apply_prefix_trim(norm_ref, ref_trim)
            candidate_hyp = apply_prefix_trim(norm_hyp, hyp_trim)
            if not candidate_ref or not candidate_hyp:
                continue

            candidate_diag = build_error_diagnostics(candidate_ref, candidate_hyp)
            candidate_wer = candidate_diag["wer"]
            candidate_prefix_match = prefix_match_count(candidate_ref.split(), candidate_hyp.split())

            better_wer = candidate_wer < best["wer"]
            same_wer_better_prefix = abs(candidate_wer - best["wer"]) < 1e-12 and candidate_prefix_match > best["prefix_match"]
            if better_wer or same_wer_better_prefix:
                best = {
                    "ref_trim": ref_trim,
                    "hyp_trim": hyp_trim,
                    "wer": candidate_wer,
                    "prefix_match": candidate_prefix_match,
                }

    improvement = base_wer - best["wer"]
    use_compensation = (
        (best["ref_trim"] > 0 or best["hyp_trim"] > 0)
        and improvement >= OFFSET_MIN_IMPROVEMENT
        and best["prefix_match"] >= base_prefix_match
    )

    return {
        "use": use_compensation,
        "base_wer": base_wer,
        "comp_wer": best["wer"],
        "ref_trim": best["ref_trim"],
        "hyp_trim": best["hyp_trim"],
        "improvement": improvement,
        "base_prefix_match": base_prefix_match,
        "comp_prefix_match": best["prefix_match"],
    }


def write_diagnostics_report(
    path,
    diagnostics,
    source_json=None,
    gt_file=None,
    section_title=None,
    metadata=None,
    wer_value=None,
    reference_word_count=None,
    hypothesis_word_count=None,
    warnings=None,
    reference_head=None,
    hypothesis_head=None,
    reference_tail=None,
    hypothesis_tail=None,
    wer_comp_value=None,
    offset_info=None,
    word_changes_info=None,
    evaluation_mode=None,
    append=False,
):
    lines = [section_title or "ASR Evaluation Report", "=" * 80]

    if source_json:
        lines.append(f"Source JSON: {source_json}")
    if gt_file:
        lines.append(f"Reference (GT): {gt_file}")
    if isinstance(metadata, dict) and metadata:
        lines.append("Mode: {mode} | Model: {model} | Backend: {backend}".format(
            mode=metadata.get("mode", "unknown"),
            model=metadata.get("model", "unknown"),
            backend=metadata.get("backend", "unknown"),
        ))
    if evaluation_mode:
        lines.append(f"Evaluation source: {evaluation_mode}")
    if isinstance(metadata, dict) and metadata:
        lines.append("Range: start={start} | end={end} | chunk={chunk}s | overlap={overlap}s".format(
            start=metadata.get("start_seconds", "None"),
            end=metadata.get("end_seconds", "None"),
            chunk=metadata.get("chunk_seconds", "None"),
            overlap=metadata.get("chunk_overlap_seconds", "None"),
        ))
        lines.append("Runtime: {runtime}s | Run started: {started} | Run finished: {finished}".format(
            runtime=metadata.get("runtime_seconds", "n/a"),
            started=metadata.get("run_started_utc", "n/a"),
            finished=metadata.get("run_finished_utc", "n/a"),
        ))

    lines.append("-" * 80)
    if wer_value is not None:
        lines.append(f"WER: {wer_value * 100:.2f} %")
    if wer_comp_value is not None:
        lines.append(f"WER (OFFSET-COMP): {wer_comp_value * 100:.2f} %")

    if isinstance(offset_info, dict):
        lines.append(
            "Offset compensation: use={use} | ref_trim={ref_trim} | hyp_trim={hyp_trim} | improvement={impr:.2f} p.b.".format(
                use=offset_info.get("use"),
                ref_trim=offset_info.get("ref_trim"),
                hyp_trim=offset_info.get("hyp_trim"),
                impr=offset_info.get("improvement", 0.0) * 100,
            )
        )

    if reference_word_count is not None and hypothesis_word_count is not None:
        lines.append(f"Word counts: REF={reference_word_count} | HYP={hypothesis_word_count}")

    if isinstance(word_changes_info, dict):
        runtime = word_changes_info.get("runtime") if isinstance(word_changes_info.get("runtime"), dict) else {}
        lines.append(
            "WordChanges: enabled={enabled} | applied_to={applied_to} | entries={entries} | checked_sub_spans={checked} | replaced_tokens={replaced}".format(
                enabled=word_changes_info.get("enabled", False),
                applied_to=word_changes_info.get("applied_to", "none"),
                entries=word_changes_info.get("entries", 0),
                checked=runtime.get("substitute_spans_checked", 0),
                replaced=runtime.get("tokens_replaced", 0),
            )
        )
        wer_before = runtime.get("wer_before")
        wer_after = runtime.get("wer_after")
        if wer_before is not None and wer_after is not None:
            lines.append(
                "WordChanges impact (WER): before={before:.2f} % | after={after:.2f} % | delta={delta:+.2f} p.b.".format(
                    before=wer_before * 100,
                    after=wer_after * 100,
                    delta=(wer_after - wer_before) * 100,
                )
            )

    lines.append(
        f"HITS: {diagnostics['hits']} | SUB: {diagnostics['substitutions']} | INS: {diagnostics['insertions']} | DEL: {diagnostics['deletions']}"
    )

    if warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in warnings)

    lines.extend(["", "Kvalitativní ukázky (2-3):"])
    examples = build_qualitative_examples(diagnostics, max_examples=3)
    if examples:
        lines.extend(f"{idx}. {item}" for idx, item in enumerate(examples, start=1))
    else:
        lines.append("(Nenalezeny žádné ukázky)")

    lines.extend([
        "",
        "Začátek srovnání:",
        f"REF: {reference_head or ''}",
        f"HYP: {hypothesis_head or ''}",
        "",
        "Konec srovnání:",
        f"REF: {reference_tail or ''}",
        f"HYP: {hypothesis_tail or ''}",
        "",
        "Top substitutions:",
    ])
    lines.extend(f"{count}x | {left} => {right}" for (left, right), count in diagnostics["top_substitutions"])

    lines.extend(["", "Top deletions:"])
    lines.extend(f"{count}x | {token}" for token, count in diagnostics["top_deletions"])

    lines.extend(["", "Top insertions:"])
    lines.extend(f"{count}x | {token}" for token, count in diagnostics["top_insertions"])

    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as file_handle:
        if append:
            file_handle.write("\n" + "=" * 80 + "\n" + "=" * 80 + "\n\n")
        file_handle.write("\n".join(lines) + "\n")


def resolve_eval_window(metadata):
    start_sec = metadata.get("start_seconds") if isinstance(metadata, dict) else None
    end_sec = metadata.get("end_seconds") if isinstance(metadata, dict) else None
    if start_sec is None and end_sec is None and isinstance(metadata, dict):
        end_sec = metadata.get("limit_seconds")
    return start_sec, end_sec


def is_segment_like(value):
    if not isinstance(value, dict) or "text" not in value:
        return False
    return any(key in value for key in ("id", "start", "end", "speakers", "is_overlap"))


def flatten_segments(data):
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    if not isinstance(data, dict):
        return []

    if isinstance(data.get("segments"), list):
        return [item for item in data.get("segments", []) if isinstance(item, dict)]
    if isinstance(data.get("data"), list):
        return [item for item in data.get("data", []) if isinstance(item, dict)]

    recording_keys = [key for key in data.keys() if re.match(r"^\d+(?:_\d+)?$", str(key))]
    if not recording_keys:
        return []

    def sort_key(key):
        match = re.match(r"^(\d+)(?:_(\d+))?$", str(key))
        if not match:
            return (10**9, 10**9)
        major = int(match.group(1))
        minor = int(match.group(2)) if match.group(2) is not None else -1
        return (major, minor)

    segments = []
    for recording in sorted(recording_keys, key=sort_key):
        value = data.get(recording)
        if isinstance(value, list):
            segments.extend(item for item in value if isinstance(item, dict))
            continue
        if isinstance(value, dict):
            if is_segment_like(value):
                segments.append(value)
                continue
            for nested_value in value.values():
                if isinstance(nested_value, list):
                    segments.extend(item for item in nested_value if isinstance(item, dict))
                elif isinstance(nested_value, dict) and is_segment_like(nested_value):
                    segments.append(nested_value)
    return segments


def filter_segments_by_recording_id(segments, recording_id=None):
    if not recording_id:
        return list(segments)

    prefix = f"seg_{recording_id}_"
    filtered = [segment for segment in segments if str(segment.get("id", "")).startswith(prefix)]
    return filtered if filtered else list(segments)


def filter_out_full_transcript_segments(segments):
    return [segment for segment in segments if not re.fullmatch(r"full_.+_transcript", str(segment.get("id", "")))]


def segment_in_window(segment, start_sec=None, end_sec=None):
    if start_sec is None and end_sec is None:
        return True
    if not isinstance(segment, dict):
        return False

    start = segment.get("start")
    end = segment.get("end")

    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        if start_sec is not None and end <= start_sec:
            return False
        if end_sec is not None and start >= end_sec:
            return False
        return True

    if isinstance(start, (int, float)):
        if start_sec is not None and start < start_sec:
            return False
        if end_sec is not None and start >= end_sec:
            return False
        return True

    return True


def clip_text_by_time_ratio(text, start, end, start_sec=None, end_sec=None):
    if not text:
        return ""
    if start_sec is None and end_sec is None:
        return text
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return text
    if end <= start:
        return text

    effective_start = start_sec if start_sec is not None else start
    effective_end = end_sec if end_sec is not None else end
    if effective_end <= effective_start:
        return ""

    intersection_start = max(start, effective_start)
    intersection_end = min(end, effective_end)
    in_range_duration = intersection_end - intersection_start
    if in_range_duration <= 0:
        return ""

    total_duration = end - start
    ratio = max(0.0, min(1.0, in_range_duration / total_duration))
    if ratio >= 0.98 or (start_sec is None and end_sec is not None and end <= end_sec + MAX_WINDOW_OVERFLOW_SECONDS):
        return text
    if ratio <= 0.02:
        return ""

    words = text.split()
    if not words:
        return ""

    left_clip = max(0.0, intersection_start - start) / total_duration
    right_clip = max(0.0, end - intersection_end) / total_duration

    start_index = int(round(len(words) * left_clip))
    end_index = int(round(len(words) * (1.0 - right_clip)))
    start_index = max(0, min(start_index, len(words) - 1))
    end_index = max(start_index + 1, min(end_index, len(words)))
    return " ".join(words[start_index:end_index]).strip()


def filter_segments_by_window(segments, start_sec=None, end_sec=None):
    if start_sec is None and end_sec is None:
        return list(segments)
    return [segment for segment in segments if segment_in_window(segment, start_sec=start_sec, end_sec=end_sec)]


def text_from_segments(segments, start_sec=None, end_sec=None):
    selected = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue

        text = clean_text(segment.get("text", ""))
        if not text:
            continue
        if not segment_in_window(segment, start_sec=start_sec, end_sec=end_sec):
            continue

        if start_sec is not None or end_sec is not None:
            text = clip_text_by_time_ratio(
                text,
                segment.get("start"),
                segment.get("end"),
                start_sec=start_sec,
                end_sec=end_sec,
            )
            if not text:
                continue

        selected.append(text)

    return " ".join(selected)


def extract_text(data, start_sec=None, end_sec=None):
    if isinstance(data, dict) and isinstance(data.get("full_transcription"), str) and data.get("full_transcription").strip():
        if isinstance(data.get("segments"), list) and data.get("segments"):
            return text_from_segments(data.get("segments", []), start_sec=start_sec, end_sec=end_sec)
        return data.get("full_transcription", "")

    segments = flatten_segments(data)
    return text_from_segments(segments, start_sec=start_sec, end_sec=end_sec) if segments else ""


def extract_full_reference_text(data, recording_id=None):
    segments = flatten_segments(data)
    if not segments:
        return "", None

    preferred_id = f"full_{recording_id}_transcript" if recording_id else None
    fallback_segment = None

    for segment in segments:
        segment_id = clean_text(segment.get("id", ""))
        if preferred_id and segment_id == preferred_id:
            return clean_text(segment.get("text", "")), segment_id
        if fallback_segment is None and re.fullmatch(r"full_.+_transcript", segment_id):
            fallback_segment = segment

    if fallback_segment is not None:
        return clean_text(fallback_segment.get("text", "")), fallback_segment.get("id")

    return "", None


def evaluate_against_single_gt(hyp_file, hyp_data, metadata, hyp_recording_id, gt_file):
    if not os.path.exists(gt_file):
        print(f"Chyba: Referenční soubor nebyl nalezen: {gt_file}")
        return None

    gt_data = load_json(gt_file)
    gt_file_name = Path(gt_file).name.lower()
    full_mode = is_full_hypothesis_file(hyp_file)

    start_sec, end_sec = resolve_eval_window(metadata)
    if full_mode:
        start_sec, end_sec = None, None

    if isinstance(gt_data, dict) and hyp_recording_id and hyp_recording_id in gt_data:
        gt_selected_data = gt_data[hyp_recording_id]
    else:
        gt_selected_data = gt_data

    ref_segments = flatten_segments(gt_selected_data)
    if not full_mode:
        ref_segments = filter_out_full_transcript_segments(ref_segments)
    ref_segments = filter_segments_by_recording_id(ref_segments, recording_id=hyp_recording_id)
    ref_segments_limited = filter_segments_by_window(ref_segments, start_sec=start_sec, end_sec=end_sec)

    hyp_segments = flatten_segments(hyp_data)
    hyp_segments_limited = filter_segments_by_window(hyp_segments, start_sec=start_sec, end_sec=end_sec)

    print("\n" + "=" * 80)
    print(f"DEBUG EVALUACE: {hyp_file}")
    print(f"Reference (GT): {gt_file}")
    print(f"Recording ID z názvu HYP: {hyp_recording_id}")
    print(f"Mód: {metadata.get('mode', 'unknown')} | Start: {start_sec} | End: {end_sec}")
    print("=" * 80)

    print(f"INFO: Počet segmentů v Reference (GT): {len(ref_segments_limited)}")

    reference_full = ""
    full_reference_id = None
    if full_mode and gt_file_name == "ground_truth_eval.json":
        reference_full, full_reference_id = extract_full_reference_text(gt_selected_data, recording_id=hyp_recording_id)
        if reference_full:
            print(f"INFO: Full-mode reference použita: {full_reference_id}")

    if not reference_full:
        reference_full = (
            text_from_segments(ref_segments_limited, start_sec=start_sec, end_sec=end_sec)
            if ref_segments
            else extract_text(gt_selected_data, start_sec=start_sec, end_sec=end_sec)
        )

    if full_reference_id:
        evaluation_mode = f"full_transcript ({full_reference_id})"
    elif full_mode:
        evaluation_mode = "full_transcript (fallback)"
    else:
        evaluation_mode = "segment_window"

    print(f"INFO: Počet segmentů v Hypotéze (ASR): {len(hyp_segments_limited)}")
    if full_mode:
        hypothesis_full = extract_text(hyp_data, start_sec=None, end_sec=None)
    else:
        hypothesis_full = (
            text_from_segments(hyp_segments_limited, start_sec=start_sec, end_sec=end_sec)
            if hyp_segments
            else extract_text(hyp_data, start_sec=start_sec, end_sec=end_sec)
        )

    if not reference_full.strip() or not hypothesis_full.strip():
        print("Chyba: Nepodařilo se extrahovat text z GT nebo HYP souboru.")
        return None

    apply_word_changes = gt_file_name in {"ground_truth_eval.json", "formal_ground_truth_eval.json"}
    word_changes_map = load_word_changes_map() if apply_word_changes else {}
    word_changes_count = len(word_changes_map)

    word_changes_runtime = {
        "substitute_spans_checked": 0,
        "tokens_replaced": 0,
        "wer_before": None,
        "wer_after": None,
    }
    word_changes_info = {
        "enabled": apply_word_changes and word_changes_count > 0,
        "applied_to": "hyp_substitutions" if apply_word_changes else "none",
        "entries": word_changes_count,
        "runtime": word_changes_runtime,
    }

    norm_ref = normalize_text(reference_full)
    norm_hyp = normalize_text(hypothesis_full)

    base_diagnostics = build_error_diagnostics(norm_ref, norm_hyp)
    final_diagnostics = base_diagnostics

    if word_changes_info["enabled"]:
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
    print(f"INFO: Počet slov v Reference: {len(ref_words)}")
    print(f"INFO: Počet slov v Hypotéze: {len(hyp_words)}")

    warnings = []
    if len(hyp_words) > len(ref_words) * 1.5:
        warning = "Hypotéza má výrazně více slov! (Možná duplicity ze sterea?)"
        warnings.append(warning)
        print(f"!!! VAROVÁNÍ: {warning}")
    elif len(hyp_words) < len(ref_words) * 0.5:
        warning = "Hypotéza má výrazně méně slov! (Možná Whisper vynechal části?)"
        warnings.append(warning)
        print(f"!!! VAROVÁNÍ: {warning}")

    if not norm_ref:
        print("Chyba: Referenční text je prázdný!")
        return None

    error_rate = final_diagnostics["wer"]
    offset_info = find_best_offset_compensation(norm_ref, norm_hyp)

    wer_comp = None
    if offset_info["use"]:
        ref_trim = offset_info["ref_trim"]
        hyp_trim = offset_info["hyp_trim"]
        norm_ref_comp = apply_prefix_trim(norm_ref, ref_trim)
        norm_hyp_comp = apply_prefix_trim(norm_hyp, hyp_trim)
        if norm_ref_comp and norm_hyp_comp:
            wer_comp = build_error_diagnostics(norm_ref_comp, norm_hyp_comp)["wer"]

    print("-" * 80)
    if word_changes_runtime["wer_before"] is not None:
        print(f"WER (BASE): {word_changes_runtime['wer_before'] * 100:.2f} %")
    print(f"WER: {error_rate * 100:.2f} %")
    if wer_comp is not None:
        print(f"WER (OFFSET-COMP): {wer_comp * 100:.2f} %")
    print("-" * 80)

    print("\nZAČÁTEK SROVNÁNÍ:")
    print(f"REF: {' '.join(ref_words[:15])}...")
    print(f"HYP: {' '.join(hyp_words[:15])}...")

    print("\nKONEC SROVNÁNÍ:")
    print(f"REF: ...{' '.join(ref_words[-15:])}")
    print(f"HYP: ...{' '.join(hyp_words[-15:])}")

    return {
        "gt_file": gt_file,
        "diagnostics": final_diagnostics,
        "wer": error_rate,
        "wer_comp": wer_comp,
        "offset_info": offset_info,
        "warnings": warnings,
        "reference_word_count": len(ref_words),
        "hypothesis_word_count": len(hyp_words),
        "reference_head": " ".join(ref_words[:15]) + "...",
        "hypothesis_head": " ".join(hyp_words[:15]) + "...",
        "reference_tail": "..." + " ".join(ref_words[-15:]),
        "hypothesis_tail": "..." + " ".join(hyp_words[-15:]),
        "word_changes_info": word_changes_info,
        "evaluation_mode": evaluation_mode,
    }


def evaluate(hyp_file):
    if not os.path.exists(hyp_file):
        print(f"Chyba: Soubor {hyp_file} nebyl nalezen!")
        return

    hyp_recording_id = extract_recording_id_from_filename(hyp_file)
    hyp_data = load_json(hyp_file)
    metadata = hyp_data.get("metadata", {}) if isinstance(hyp_data, dict) else {}
    diagnostics_path = diagnostics_output_path(hyp_file)

    gt_files = [
        "results/ground_truth_eval.json",
        "results/formal_ground_truth_eval.json",
    ]
    existing_gt_files = [path for path in gt_files if os.path.exists(path)]
    if not existing_gt_files:
        print("Chyba: Nebyl nalezen ani jeden GT eval soubor.")
        print("Očekáváno: results/ground_truth_eval.json a results/formal_ground_truth_eval.json")
        return

    first_section = True
    for gt_file in existing_gt_files:
        section_result = evaluate_against_single_gt(
            hyp_file=hyp_file,
            hyp_data=hyp_data,
            metadata=metadata,
            hyp_recording_id=hyp_recording_id,
            gt_file=gt_file,
        )
        if section_result is None:
            continue

        write_diagnostics_report(
            diagnostics_path,
            section_result["diagnostics"],
            source_json=hyp_file,
            gt_file=gt_file,
            section_title=f"ASR Evaluation Report | {Path(gt_file).name}",
            metadata=metadata,
            evaluation_mode=section_result.get("evaluation_mode"),
            wer_value=section_result["wer"],
            reference_word_count=section_result["reference_word_count"],
            hypothesis_word_count=section_result["hypothesis_word_count"],
            warnings=section_result["warnings"],
            reference_head=section_result["reference_head"],
            hypothesis_head=section_result["hypothesis_head"],
            reference_tail=section_result["reference_tail"],
            hypothesis_tail=section_result["hypothesis_tail"],
            wer_comp_value=section_result["wer_comp"],
            offset_info=section_result["offset_info"],
            word_changes_info=section_result.get("word_changes_info"),
            append=not first_section,
        )
        first_section = False

        print("=" * 80)
        print("=" * 80)

    if not first_section:
        print(f"DEBUG: Kompletní report uložen do {diagnostics_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Použití: python scripts/evaluate_wer.py results/vysledek.json")
    else:
        evaluate(sys.argv[1])
