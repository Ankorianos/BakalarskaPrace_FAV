# -*- coding: utf-8 -*-
import argparse
import importlib
import sys
from pathlib import Path

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = (PROJECT_ROOT.parent / "INTERSPEECH2023").resolve()
DEFAULT_LM_PATH = (DEFAULT_MODEL_DIR / "LM.arpa").resolve()

REQUIRED_FILES = [
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "pytorch_model.bin",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Jednoduchý test načtení lokálního Wav2Vec2 modelu."
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help=f"Složka s modelem (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--lm-path",
        default=str(DEFAULT_LM_PATH),
        help=f"Cesta k LM.arpa (default: {DEFAULT_LM_PATH})",
    )
    return parser.parse_args()


def check_required_files(model_dir: Path):
    missing = []
    for name in REQUIRED_FILES:
        if not (model_dir / name).exists():
            missing.append(name)
    return missing


def load_unigrams_utf8(arpa_path: Path):
    text = arpa_path.read_text(encoding="utf-8")
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


def test_lm_load(processor, lm_path: Path):
    if not lm_path.exists():
        return False, f"LM soubor neexistuje: {lm_path}"

    try:
        pyctcdecode_module = importlib.import_module("pyctcdecode")
        build_ctcdecoder = pyctcdecode_module.build_ctcdecoder
    except Exception as exc:
        return False, f"LM test přeskočen: pyctcdecode není dostupné ({exc})"

    vocab_dict = processor.tokenizer.get_vocab()
    sorted_vocab = [token for token, _ in sorted(vocab_dict.items(), key=lambda item: item[1])]

    try:
        _decoder = build_ctcdecoder(
            labels=sorted_vocab,
            kenlm_model_path=str(lm_path),
            unigrams=[],
        )
        return True, f"LM decoder načten přes pyctcdecode (unigrams=[]) : {lm_path}"
    except Exception as exc:
        first_error = str(exc)

    try:
        unigrams = load_unigrams_utf8(lm_path)
        _decoder = build_ctcdecoder(
            labels=sorted_vocab,
            kenlm_model_path=str(lm_path),
            unigrams=unigrams,
        )
        return True, f"LM decoder načten přes pyctcdecode (UTF-8 unigrams={len(unigrams)}) : {lm_path}"
    except Exception as exc:
        return (
            False,
            "LM decoder přes pyctcdecode selhal. "
            f"Pokus1(unigrams=[]): {first_error} | Pokus2(UTF-8 unigrams): {exc}",
        )


def main():
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    lm_path = Path(args.lm_path).expanduser().resolve() if args.lm_path else None

    print("=" * 80)
    print("Wav2Vec2 load check")
    print(f"Model dir: {model_dir}")
    print(f"LM path:   {lm_path}")

    if not model_dir.exists():
        print(f"ERROR: Model složka neexistuje: {model_dir}")
        sys.exit(1)

    missing = check_required_files(model_dir)
    if missing:
        print("ERROR: Chybí soubory modelu:")
        for item in missing:
            print(f"- {item}")
        sys.exit(1)

    try:
        processor = Wav2Vec2Processor.from_pretrained(str(model_dir), local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(str(model_dir), local_files_only=True)
    except Exception as exc:
        print(f"ERROR: Načtení modelu selhalo: {exc}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    with torch.no_grad():
        dummy_audio = torch.zeros((1, 16000), dtype=torch.float32)
        inputs = processor(dummy_audio.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None
        logits = model(input_values, attention_mask=attention_mask).logits

    total_params = sum(parameter.numel() for parameter in model.parameters())

    print("OK: Processor načten")
    print("OK: Model načten")
    print(f"OK: Forward pass proběhl | logits shape = {tuple(logits.shape)}")
    print(f"Info: device={device} | params={total_params}")

    if lm_path is not None:
        lm_ok, lm_msg = test_lm_load(processor, lm_path)
        status = "OK" if lm_ok else "WARN"
        print(f"{status}: {lm_msg}")

    print("=" * 80)


if __name__ == "__main__":
    main()
