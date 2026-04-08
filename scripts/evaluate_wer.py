import json
import sys
import os
from pathlib import Path
from collections import Counter
from jiwer import wer, process_words
import re
import unicodedata

# Seznam známých halucinací Whisperu
HALLUCINATIONS = [
    r"titulky vytvořil.*",
    r"děkuji za sledování.*",
    r"odběratelé.*",
    r"přeložil.*",
    r"reaping.*",
    r"watch next.*",
    r"thanks for watching.*",
    r"titulky.*"
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
    "devadesát", "devadesat", "stě", "ste", "sto", "set", "tisíc", "tisic"
}

NUMBER_STEMS_ASCII = {
    "nul", "jedn", "dva", "dv", "tri", "ctyr", "pet", "sest", "sedm", "osm", "devet",
    "deset", "jedenact", "dvanact", "trinact", "ctrnact", "patnact", "sestnact", "sedmnact",
    "osmnact", "devatenact", "devatenacet", "dvacet", "tricet", "ctyricet", "padesat", "sedesat",
    "sedmdesat", "osmdesat", "devadesat", "sto", "ste", "set", "tisic"
}

NUMBER_SUFFIXES_ASCII = (
    "eho", "emu", "em", "ym", "ych", "ymi", "y", "a", "u", "ou", "i", "o", "ho",
    "teho", "ateho", "cateho", "nacteho"
)

MAX_WINDOW_OVERFLOW_SECONDS = 0.6
OFFSET_MAX_TRIM_WORDS = 6
OFFSET_MIN_IMPROVEMENT = 0.01
OFFSET_PREFIX_WINDOW = 12

COLLOQUIAL_TOKEN_MAP = {
    "jmenuju": "jmenuji",
    "řikám": "říkám",
    "řikal": "říkal",
    "řikala": "říkala",
    "řikali": "říkali",
    "vono": "ono",
    "von": "on",
    "voni": "oni",
    "kerej": "který",
    "kerý": "který",
    "kerá": "která",
    "kerou": "kterou",
    "nevim": "nevím",
    "vim": "vím",
    "bejval": "býval",
    "bejvám": "bývám",
}


def strip_diacritics(text):
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


NUMBER_WORDS_ASCII = {strip_diacritics(word) for word in NUMBER_WORDS}


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
            stem = token_ascii[: -len(suffix)]
            if len(stem) >= 3 and stem in NUMBER_STEMS_ASCII:
                return True

    return False


def normalize_common(text):
    normalized = (text or "").lower()
    for pattern in HALLUCINATIONS:
        normalized = re.sub(pattern, " ", normalized, flags=re.IGNORECASE)
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def collapse_letter_spelling(tokens):
    mapped = [LETTER_NAME_MAP.get(token, token) for token in tokens]
    output = []
    index = 0
    while index < len(mapped):
        token = mapped[index]
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

        output.append(token)
        index += 1

    return output


def normalize_numbers_to_placeholder(tokens):
    output = []
    for token in tokens:
        if is_numeric_like_token(token):
            output.append("<num>")
        else:
            output.append(token)

    compacted = []
    for token in output:
        if token == "<num>" and compacted and compacted[-1] == "<num>":
            continue
        compacted.append(token)
    return compacted


def normalize_colloquial_variants(tokens):
    normalized_tokens = []
    for token in tokens:
        mapped = COLLOQUIAL_TOKEN_MAP.get(token, token)

        if mapped.endswith("uju") and len(mapped) > 4:
            mapped = mapped[:-3] + "ji"

        normalized_tokens.append(mapped)

    return normalized_tokens


def normalize_text_strict(text):
    normalized = normalize_common(text)
    return " ".join(normalized.split()).strip()


def normalize_text_robust(text):
    normalized = normalize_common(text)
    tokens = normalized.split()
    tokens = collapse_letter_spelling(tokens)
    tokens = normalize_colloquial_variants(tokens)
    tokens = normalize_numbers_to_placeholder(tokens)
    return " ".join(tokens).strip()


def prefix_match_count(left_tokens, right_tokens, window=OFFSET_PREFIX_WINDOW):
    compare_len = min(window, len(left_tokens), len(right_tokens))
    if compare_len <= 0:
        return 0
    return sum(1 for idx in range(compare_len) if left_tokens[idx] == right_tokens[idx])


def apply_prefix_trim(text, trim_count):
    tokens = (text or "").split()
    if trim_count <= 0:
        return " ".join(tokens).strip()
    return " ".join(tokens[trim_count:]).strip()


def find_best_offset_compensation(norm_ref_robust, norm_hyp_robust):
    ref_tokens = norm_ref_robust.split()
    hyp_tokens = norm_hyp_robust.split()

    baseline = build_error_diagnostics(norm_ref_robust, norm_hyp_robust)
    base_wer = baseline["wer"]
    base_prefix_match = prefix_match_count(ref_tokens, hyp_tokens)

    best = {
        "ref_trim": 0,
        "hyp_trim": 0,
        "wer": base_wer,
        "prefix_match": base_prefix_match,
    }

    for trim in range(1, OFFSET_MAX_TRIM_WORDS + 1):
        candidates = [(trim, 0), (0, trim)]

        for ref_trim, hyp_trim in candidates:
            candidate_ref = apply_prefix_trim(norm_ref_robust, ref_trim)
            candidate_hyp = apply_prefix_trim(norm_hyp_robust, hyp_trim)

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
                left = " ".join(ref_words[change.ref_start_idx : change.ref_end_idx])
                right = " ".join(hyp_words[change.hyp_start_idx : change.hyp_end_idx])
                substitutions[(left, right)] += 1
            elif change.type == "insert":
                token = " ".join(hyp_words[change.hyp_start_idx : change.hyp_end_idx])
                if token:
                    insertions[token] += 1
            elif change.type == "delete":
                token = " ".join(ref_words[change.ref_start_idx : change.ref_end_idx])
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
        if len(examples) >= max_examples:
            return examples

    for token, count in diagnostics.get("top_deletions", []):
        if token:
            examples.append(f"DEL ({count}x): '{token}'")
        if len(examples) >= max_examples:
            return examples

    for token, count in diagnostics.get("top_insertions", []):
        if token:
            examples.append(f"INS ({count}x): '{token}'")
        if len(examples) >= max_examples:
            return examples

    return examples


def write_diagnostics_report(
    path,
    diagnostics,
    source_json=None,
    gt_file=None,
    section_title=None,
    metadata=None,
    strict_wer=None,
    robust_wer=None,
    reference_word_count=None,
    hypothesis_word_count=None,
    warnings=None,
    reference_head=None,
    hypothesis_head=None,
    reference_tail=None,
    hypothesis_tail=None,
    strict_comp_wer=None,
    robust_comp_wer=None,
    offset_info=None,
    append=False,
):
    lines = []
    lines.append(section_title or "ASR Evaluation Report")
    lines.append("=" * 80)

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

    if strict_wer is not None:
        lines.append(f"WER (STRICT): {strict_wer * 100:.2f} %")
    if robust_wer is not None:
        lines.append(f"WER (ROBUST): {robust_wer * 100:.2f} %")
    if strict_comp_wer is not None:
        lines.append(f"WER (STRICT, OFFSET-COMP): {strict_comp_wer * 100:.2f} %")
    if robust_comp_wer is not None:
        lines.append(f"WER (ROBUST, OFFSET-COMP): {robust_comp_wer * 100:.2f} %")

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

    lines.append(
        f"HITS: {diagnostics['hits']} | SUB: {diagnostics['substitutions']} | INS: {diagnostics['insertions']} | DEL: {diagnostics['deletions']}"
    )

    if warnings:
        lines.append("")
        lines.append("Warnings:")
        for warning in warnings:
            lines.append(f"- {warning}")

    lines.append("")
    lines.append("Kvalitativní ukázky (2-3):")
    examples = build_qualitative_examples(diagnostics, max_examples=3)
    if examples:
        for idx, item in enumerate(examples, start=1):
            lines.append(f"{idx}. {item}")
    else:
        lines.append("(Nenalezeny žádné ukázky)")

    lines.append("")
    lines.append("Začátek srovnání:")
    lines.append(f"REF: {reference_head or ''}")
    lines.append(f"HYP: {hypothesis_head or ''}")

    lines.append("")
    lines.append("Konec srovnání:")
    lines.append(f"REF: {reference_tail or ''}")
    lines.append(f"HYP: {hypothesis_tail or ''}")

    lines.append("")
    lines.append("Top substitutions:")
    for (left, right), count in diagnostics["top_substitutions"]:
        lines.append(f"{count}x | {left} => {right}")
    lines.append("")
    lines.append("Top deletions:")
    for token, count in diagnostics["top_deletions"]:
        lines.append(f"{count}x | {token}")
    lines.append("")
    lines.append("Top insertions:")
    for token, count in diagnostics["top_insertions"]:
        lines.append(f"{count}x | {token}")

    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as file_handle:
        if append:
            file_handle.write("\n" + "=" * 80 + "\n" + "=" * 80 + "\n\n")
        file_handle.write("\n".join(lines) + "\n")


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


def resolve_eval_window(metadata):
    start_sec = metadata.get("start_seconds") if isinstance(metadata, dict) else None
    end_sec = metadata.get("end_seconds") if isinstance(metadata, dict) else None

    if start_sec is None and end_sec is None and isinstance(metadata, dict):
        end_sec = metadata.get("limit_seconds")

    return start_sec, end_sec


def segment_in_window(segment, start_sec=None, end_sec=None):
    if start_sec is None and end_sec is None:
        return True

    if not isinstance(segment, dict):
        return False

    start = segment.get("start")
    end = segment.get("end")

    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        if end <= start_sec if start_sec is not None else False:
            return False
        if start >= end_sec if end_sec is not None else False:
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


def load_json(path):
    with open(path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def is_segment_like(value):
    if not isinstance(value, dict):
        return False
    if "text" not in value:
        return False
    return any(key in value for key in ("id", "start", "end", "speakers", "is_overlap"))


def recording_map_to_segments(data):
    if not isinstance(data, dict):
        return []

    def is_recording_key(key):
        return bool(re.match(r"^\d+(?:_\d+)?$", str(key)))

    def recording_sort_key(key):
        key_str = str(key)
        match = re.match(r"^(\d+)(?:_(\d+))?$", key_str)
        if not match:
            return (10**9, 10**9)
        major = int(match.group(1))
        minor = int(match.group(2)) if match.group(2) is not None else -1
        return (major, minor)

    recording_keys = [key for key in data.keys() if is_recording_key(key)]
    if not recording_keys:
        return []

    segments = []

    for recording in sorted(recording_keys, key=recording_sort_key):
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


def segments_from_data(data):
    if isinstance(data, dict):
        if isinstance(data.get("segments"), list):
            return data.get("segments", [])
        if isinstance(data.get("data"), list):
            return data.get("data", [])
        mapped_segments = recording_map_to_segments(data)
        if mapped_segments:
            return mapped_segments
        return []
    if isinstance(data, list):
        return data
    return []


def filter_segments_by_recording_id(segments, recording_id=None):
    if not recording_id:
        return list(segments)

    filtered = []
    prefix = f"seg_{recording_id}_"

    for segment in segments:
        if not isinstance(segment, dict):
            continue

        segment_id = str(segment.get("id", ""))
        if segment_id.startswith(prefix):
            filtered.append(segment)

    if filtered:
        return filtered

    return list(segments)


def text_from_segments(segments, start_sec=None, end_sec=None):
    selected = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        text = segment.get("text", "")
        if not text:
            continue

        if not segment_in_window(segment, start_sec=start_sec, end_sec=end_sec):
            continue

        if start_sec is not None or end_sec is not None:
            text = clip_text_by_time_ratio(
                str(text),
                segment.get("start"),
                segment.get("end"),
                start_sec=start_sec,
                end_sec=end_sec,
            )
            if not text:
                continue

        selected.append(str(text))
    return " ".join(selected)


def extract_text(data, start_sec=None, end_sec=None):
    if isinstance(data, dict):
        if isinstance(data.get("full_transcription"), str) and data.get("full_transcription").strip():
            if isinstance(data.get("segments"), list) and data.get("segments"):
                return text_from_segments(data.get("segments", []), start_sec=start_sec, end_sec=end_sec)
            return data.get("full_transcription", "")

    segments = segments_from_data(data)
    if segments:
        return text_from_segments(segments, start_sec=start_sec, end_sec=end_sec)
    return ""


def filter_segments_by_window(segments, start_sec=None, end_sec=None):
    if start_sec is None and end_sec is None:
        return list(segments)

    filtered = []
    for segment in segments:
        if segment_in_window(segment, start_sec=start_sec, end_sec=end_sec):
            filtered.append(segment)

    return filtered

def resolve_gt_file(hyp_file):
    gt_eval_file = "results/ground_truth_eval.json"
    gt_raw_file = "results/ground_truth_raw.json"
    formal_gt_eval_file = "results/formal_ground_truth_eval.json"

    hyp_name = Path(hyp_file).name.lower()
    if "formal" in hyp_name:
        candidates = [
            formal_gt_eval_file,
            gt_eval_file,
            gt_raw_file,
        ]
    else:
        candidates = [
            gt_eval_file,
            gt_raw_file,
            formal_gt_eval_file,
        ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return candidates[0]


def evaluate_against_single_gt(hyp_file, hyp_data, metadata, hyp_recording_id, gt_file):
    if not os.path.exists(gt_file):
        print(f"Chyba: Referenční soubor nebyl nalezen: {gt_file}")
        return None

    gt_data = load_json(gt_file)
    start_sec, end_sec = resolve_eval_window(metadata)

    hyp_segments = segments_from_data(hyp_data)

    if isinstance(gt_data, dict) and hyp_recording_id and hyp_recording_id in gt_data:
        gt_selected_data = gt_data[hyp_recording_id]
    else:
        gt_selected_data = gt_data

    ref_segments = segments_from_data(gt_selected_data)
    ref_segments = filter_segments_by_recording_id(ref_segments, recording_id=hyp_recording_id)
    ref_segments_limited = filter_segments_by_window(ref_segments, start_sec=start_sec, end_sec=end_sec)
    hyp_segments_limited = filter_segments_by_window(hyp_segments, start_sec=start_sec, end_sec=end_sec)

    print("\n" + "="*80)
    print(f"DEBUG EVALUACE: {hyp_file}")
    print(f"Reference (GT): {gt_file}")
    print(f"Recording ID z názvu HYP: {hyp_recording_id}")
    print(f"Mód: {metadata.get('mode', 'unknown')} | Start: {start_sec} | End: {end_sec}")
    print("="*80)

    print(f"INFO: Počet segmentů v Reference (GT): {len(ref_segments_limited)}")
    if ref_segments:
        reference_full = text_from_segments(ref_segments_limited, start_sec=start_sec, end_sec=end_sec)
    else:
        reference_full = extract_text(gt_selected_data, start_sec=start_sec, end_sec=end_sec)

    print(f"INFO: Počet segmentů v Hypotéze (ASR): {len(hyp_segments_limited)}")
    if hyp_segments:
        hypothesis_full = text_from_segments(hyp_segments_limited, start_sec=start_sec, end_sec=end_sec)
    else:
        hypothesis_full = extract_text(hyp_data, start_sec=start_sec, end_sec=end_sec)

    if not reference_full.strip() or not hypothesis_full.strip():
        print("Chyba: Nepodařilo se extrahovat text z GT nebo HYP souboru.")
        return None

    norm_ref_strict = normalize_text_strict(reference_full)
    norm_hyp_strict = normalize_text_strict(hypothesis_full)
    norm_ref_robust = normalize_text_robust(reference_full)
    norm_hyp_robust = normalize_text_robust(hypothesis_full)

    ref_words = norm_ref_robust.split()
    hyp_words = norm_hyp_robust.split()

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

    diagnostics = build_error_diagnostics(norm_ref_robust, norm_hyp_robust)

    if not norm_ref_strict:
        print("Chyba: Referenční text je prázdný!")
        return None

    error_rate_strict = wer(norm_ref_strict, norm_hyp_strict)
    error_rate_robust = diagnostics["wer"]

    offset_info = find_best_offset_compensation(norm_ref_robust, norm_hyp_robust)
    strict_comp = None
    robust_comp = None
    if offset_info["use"]:
        ref_trim = offset_info["ref_trim"]
        hyp_trim = offset_info["hyp_trim"]

        norm_ref_strict_comp = apply_prefix_trim(norm_ref_strict, ref_trim)
        norm_hyp_strict_comp = apply_prefix_trim(norm_hyp_strict, hyp_trim)
        norm_ref_robust_comp = apply_prefix_trim(norm_ref_robust, ref_trim)
        norm_hyp_robust_comp = apply_prefix_trim(norm_hyp_robust, hyp_trim)

        if norm_ref_strict_comp and norm_hyp_strict_comp:
            strict_comp = wer(norm_ref_strict_comp, norm_hyp_strict_comp)
        if norm_ref_robust_comp and norm_hyp_robust_comp:
            robust_comp = build_error_diagnostics(norm_ref_robust_comp, norm_hyp_robust_comp)["wer"]

    print("-" * 80)
    print(f"WER (STRICT): {error_rate_strict * 100:.2f} %")
    print(f"WER (ROBUST): {error_rate_robust * 100:.2f} %")
    if strict_comp is not None:
        print(f"WER (STRICT, OFFSET-COMP): {strict_comp * 100:.2f} %")
    if robust_comp is not None:
        print(f"WER (ROBUST, OFFSET-COMP): {robust_comp * 100:.2f} %")
    print("-" * 80)

    print("\nZAČÁTEK SROVNÁNÍ:")
    print(f"REF: {' '.join(ref_words[:15])}...")
    print(f"HYP: {' '.join(hyp_words[:15])}...")

    print("\nKONEC SROVNÁNÍ:")
    print(f"REF: ...{' '.join(ref_words[-15:])}")
    print(f"HYP: ...{' '.join(hyp_words[-15:])}")

    return {
        "gt_file": gt_file,
        "diagnostics": diagnostics,
        "strict_wer": error_rate_strict,
        "robust_wer": error_rate_robust,
        "strict_comp": strict_comp,
        "robust_comp": robust_comp,
        "offset_info": offset_info,
        "warnings": warnings,
        "reference_word_count": len(ref_words),
        "hypothesis_word_count": len(hyp_words),
        "reference_head": " ".join(ref_words[:15]) + "...",
        "hypothesis_head": " ".join(hyp_words[:15]) + "...",
        "reference_tail": "..." + " ".join(ref_words[-15:]),
        "hypothesis_tail": "..." + " ".join(hyp_words[-15:]),
    }


def evaluate(hyp_file):
    hyp_recording_id = extract_recording_id_from_filename(hyp_file)

    if not os.path.exists(hyp_file):
        print(f"Chyba: Soubor {hyp_file} nebyl nalezen!")
        return

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
            strict_wer=section_result["strict_wer"],
            robust_wer=section_result["robust_wer"],
            reference_word_count=section_result["reference_word_count"],
            hypothesis_word_count=section_result["hypothesis_word_count"],
            warnings=section_result["warnings"],
            reference_head=section_result["reference_head"],
            hypothesis_head=section_result["hypothesis_head"],
            reference_tail=section_result["reference_tail"],
            hypothesis_tail=section_result["hypothesis_tail"],
            strict_comp_wer=section_result["strict_comp"],
            robust_comp_wer=section_result["robust_comp"],
            offset_info=section_result["offset_info"],
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
