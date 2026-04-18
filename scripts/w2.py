# -*- coding: utf-8 -*-
import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = (PROJECT_ROOT.parent / "INTERSPEECH2023").resolve()
DEFAULT_LM_PATH = (DEFAULT_MODEL_DIR / "LM.arpa").resolve()
DEFAULT_AUDIO_PATH = (PROJECT_ROOT / "Test_sentence.wav").resolve()

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
    parser.add_argument(
        "--audio-path",
        default=str(DEFAULT_AUDIO_PATH),
        help=f"Cesta k test WAV souboru (default: {DEFAULT_AUDIO_PATH})",
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


def to_float32(audio):
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = max(abs(info.min), info.max)
        return audio.astype(np.float32) / float(scale)
    return audio.astype(np.float32)


def load_audio_mono_16k(audio_path: Path):
    sample_rate, data = wavfile.read(str(audio_path))

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    audio = to_float32(data)

    if sample_rate != 16000:
        duration = len(audio) / float(sample_rate)
        target_len = int(round(duration * 16000))
        if target_len <= 1:
            return np.zeros((1,), dtype=np.float32)
        x_old = np.linspace(0.0, duration, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, duration, num=target_len, endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)

    return audio


def build_lm_decoder(processor, lm_path: Path):
    if not lm_path.exists():
        return None, f"LM soubor neexistuje: {lm_path}"

    try:
        pyctcdecode_module = importlib.import_module("pyctcdecode")
        build_ctcdecoder = pyctcdecode_module.build_ctcdecoder
    except Exception as exc:
        return None, f"LM test přeskočen: pyctcdecode není dostupné ({exc})"

    try:
        vocab_dict = processor.tokenizer.get_vocab()
        labels = [token for token, _ in sorted(vocab_dict.items(), key=lambda item: item[1])]
        unigrams = load_unigrams_utf8(lm_path)

        decoder = build_ctcdecoder(
            labels=labels,
            kenlm_model_path=str(lm_path),
            unigrams=unigrams,
        )
        return decoder, f"LM decoder načten přes pyctcdecode (UTF-8 unigrams={len(unigrams)}) : {lm_path}"
    except Exception as exc:
        return None, f"LM decoder přes pyctcdecode selhal ({exc})"


def main():
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    lm_path = Path(args.lm_path).expanduser().resolve() if args.lm_path else None
    audio_path = Path(args.audio_path).expanduser().resolve() if args.audio_path else None

    print("=" * 80)
    print("Wav2Vec2 load check")
    print(f"Model dir: {model_dir}")
    print(f"LM path:   {lm_path}")
    print(f"Audio:     {audio_path}")

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

    lm_decoder = None
    if lm_path is not None:
        lm_decoder, lm_msg = build_lm_decoder(processor, lm_path)
        status = "OK" if lm_decoder is not None else "WARN"
        print(f"{status}: {lm_msg}")

    if audio_path is not None and audio_path.exists():
        audio = load_audio_mono_16k(audio_path)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None


        with torch.no_grad():
            logits = model(input_values, attention_mask=attention_mask).logits

        pred_ids = torch.argmax(logits, dim=-1)
        greedy_text = processor.batch_decode(pred_ids)[0].strip()
        print(f"Greedy decode (test_sentence): {greedy_text}")

        if lm_decoder is not None:
            lm_text = lm_decoder.decode(logits[0].detach().cpu().numpy()).strip()
            print(f"LM decode (test_sentence): {lm_text}")
    else:
        print("WARN: test_sentence.wav nebyl nalezen, inference přes audio přeskočena.")

    print("=" * 80)


if __name__ == "__main__":
    main()
