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


def write_diagnostics_report(path, diagnostics, source_json=None):
    lines = []
    lines.append("WER Diagnostics (ROBUST)")
    lines.append("=" * 80)
    if source_json:
        lines.append(f"Source JSON: {source_json}")
    if source_json:
        lines.append("-" * 80)
    lines.append(f"WER: {diagnostics['wer'] * 100:.2f} %")
    lines.append(
        f"HITS: {diagnostics['hits']} | SUB: {diagnostics['substitutions']} | INS: {diagnostics['insertions']} | DEL: {diagnostics['deletions']}"
    )
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

    with open(path, "w", encoding="utf-8") as file_handle:
        file_handle.write("\n".join(lines) + "\n")


def diagnostics_output_path(hyp_file):
    stem = Path(hyp_file).stem
    safe_stem = re.sub(r"[^\w.-]+", "_", stem).strip("._") or "hypothesis"
    return f"results/wer_diagnostics_{safe_stem}.txt"


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


def segments_from_data(data):
    if isinstance(data, dict):
        if isinstance(data.get("segments"), list):
            return data.get("segments", [])
        if isinstance(data.get("data"), list):
            return data.get("data", [])
        return []
    if isinstance(data, list):
        return data
    return []


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

def evaluate(hyp_file):
    gt_eval_file = "results/ground_truth_eval.json"
    gt_raw_file = "results/ground_truth_raw.json"
    gt_file = gt_eval_file if os.path.exists(gt_eval_file) else gt_raw_file

    if not os.path.exists(gt_file):
        print("Chyba: Referenční soubor nebyl nalezen (čekám results/ground_truth_eval.json nebo results/ground_truth_raw.json)")
        return

    if not os.path.exists(hyp_file):
        print(f"Chyba: Soubor {hyp_file} nebyl nalezen!")
        return
    
    gt_data = load_json(gt_file)
    hyp_data = load_json(hyp_file)
    
    metadata = hyp_data.get("metadata", {}) if isinstance(hyp_data, dict) else {}
    start_sec, end_sec = resolve_eval_window(metadata)

    hyp_segments = segments_from_data(hyp_data)
    ref_segments = segments_from_data(gt_data)
    ref_segments_limited = filter_segments_by_window(ref_segments, start_sec=start_sec, end_sec=end_sec)
    hyp_segments_limited = filter_segments_by_window(hyp_segments, start_sec=start_sec, end_sec=end_sec)

    print("\n" + "="*80)
    print(f"DEBUG EVALUACE: {hyp_file}")
    print(f"Mód: {metadata.get('mode', 'unknown')} | Start: {start_sec} | End: {end_sec}")
    print("="*80)

    # 1. Příprava REFERENCE (GT)
    print(f"INFO: Počet segmentů v Reference (GT): {len(ref_segments_limited)}")
    if ref_segments:
        reference_full = text_from_segments(ref_segments_limited, start_sec=start_sec, end_sec=end_sec)
    else:
        reference_full = extract_text(gt_data, start_sec=start_sec, end_sec=end_sec)

    # 2. Příprava HYPOTÉZY (ASR)
    print(f"INFO: Počet segmentů v Hypotéze (ASR): {len(hyp_segments_limited)}")
    if hyp_segments:
        hypothesis_full = text_from_segments(hyp_segments_limited, start_sec=start_sec, end_sec=end_sec)
    else:
        hypothesis_full = extract_text(hyp_data, start_sec=start_sec, end_sec=end_sec)

    if not reference_full.strip() or not hypothesis_full.strip():
        print("Chyba: Nepodařilo se extrahovat text z GT nebo HYP souboru.")
        return

    # 3. Normalizace
    norm_ref_strict = normalize_text_strict(reference_full)
    norm_hyp_strict = normalize_text_strict(hypothesis_full)
    norm_ref_robust = normalize_text_robust(reference_full)
    norm_hyp_robust = normalize_text_robust(hypothesis_full)

    # 4. Diagnostika slov
    ref_words = norm_ref_robust.split()
    hyp_words = norm_hyp_robust.split()
    
    print(f"INFO: Počet slov v Reference: {len(ref_words)}")
    print(f"INFO: Počet slov v Hypotéze: {len(hyp_words)}")
    
    if len(hyp_words) > len(ref_words) * 1.5:
        print("!!! VAROVÁNÍ: Hypotéza má výrazně více slov! (Možná duplicity ze sterea?)")
    elif len(hyp_words) < len(ref_words) * 0.5:
        print("!!! VAROVÁNÍ: Hypotéza má výrazně méně slov! (Možná Whisper vynechal části?)")

    # 5. Uložení vyčištěných textů pro manuální kontrolu
    with open("results/ref_clean.txt", "w", encoding="utf-8") as file_handle:
        file_handle.write(norm_ref_robust)
    with open("results/hyp_clean.txt", "w", encoding="utf-8") as file_handle:
        file_handle.write(norm_hyp_robust)
    with open("results/ref_clean_strict.txt", "w", encoding="utf-8") as file_handle:
        file_handle.write(norm_ref_strict)
    with open("results/hyp_clean_strict.txt", "w", encoding="utf-8") as file_handle:
        file_handle.write(norm_hyp_strict)
    print("DEBUG: Vyčištěné texty uloženy do results/*clean*.txt")

    diagnostics = build_error_diagnostics(norm_ref_robust, norm_hyp_robust)
    diagnostics_path = diagnostics_output_path(hyp_file)
    write_diagnostics_report(
        diagnostics_path,
        diagnostics,
        source_json=hyp_file,
    )
    print(f"DEBUG: Diagnostický report uložen do {diagnostics_path}")

    # 6. Výpočet WER
    if not norm_ref_strict:
        print("Chyba: Referenční text je prázdný!")
        return

    error_rate_strict = wer(norm_ref_strict, norm_hyp_strict)
    error_rate_robust = diagnostics["wer"]

    print("-" * 80)
    print(f"WER (STRICT): {error_rate_strict * 100:.2f} %")
    print(f"WER (ROBUST): {error_rate_robust * 100:.2f} %")
    print("-" * 80)

    # Ukázka srovnání začátku a konce
    print("\nZAČÁTEK SROVNÁNÍ:")
    print(f"REF: {' '.join(ref_words[:15])}...")
    print(f"HYP: {' '.join(hyp_words[:15])}...")
    
    print("\nKONEC SROVNÁNÍ:")
    print(f"REF: ...{' '.join(ref_words[-15:])}")
    print(f"HYP: ...{' '.join(hyp_words[-15:])}")
    print("="*80 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Použití: python scripts/evaluate_wer.py results/vysledek.json")
    else:
        evaluate(sys.argv[1])
