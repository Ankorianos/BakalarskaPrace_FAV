# -*- coding: utf-8 -*-
import argparse
import importlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = (PROJECT_ROOT.parent / "INTERSPEECH2023").resolve()
DEFAULT_LM_PATH = (DEFAULT_MODEL_DIR / "LM.arpa").resolve()



def inspect_lm_file(lm_path):
    lm_file = Path(lm_path)
    print(f"--- Kontrola souboru: {lm_file.name} ---")
    
    if not lm_file.exists():
        print("CHYBA: Soubor neexistuje.")
        return

    # 1. Zkusíme zjistit, zda nejde o binární soubor (KenLM .bin)
    with open(lm_file, 'rb') as f:
        header = f.read(128)
        # KenLM binární soubory často začínají specifickou signaturou
        if b"kenlm" in header.lower():
            print("INFO: Soubor vypadá jako BINÁRNÍ KenLM formát.")
        else:
            print("INFO: Soubor se tváří jako TEXTOVÝ (ARPA).")

    # 2. Test čtení v UTF-8
    line_count = 0
    errors = 0
    
    print("Probíhá kontrola kódování UTF-8...")
    try:
        with open(lm_file, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                try:
                    # Jen simulujeme čtení řádku
                    _ = line
                except UnicodeDecodeError as e:
                    print(f"CHYBA DEKÓDOVÁNÍ na řádku {line_number}: {e}")
                    errors += 1
                    if errors >= 5: # Vypíšeme jen prvních 5 chyb
                        print("Příliš mnoho chyb, zastavuji kontrolu...")
                        break
                line_count = line_number
    except Exception as e:
        print(f"Kritická chyba při čtení: {e}")

    if errors == 0:
        print(f"Úspěch: Soubor je v pořádku v UTF-8 (zkontrolováno {line_count} řádků).")
    else:
        print(f"Celkem nalezeno chyb: {errors}")













def parse_args():
    parser = argparse.ArgumentParser(
        description="Porovnání W2V2 greedy decode vs LM decode (pyctcdecode) s podporou dummy audia."
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=None,
        help="Volitelná cesta k WAV. Když není zadáno, použije se --dummy.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help=f"Složka modelu (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--lm-path",
        default=str(DEFAULT_LM_PATH),
        help=f"Cesta k LM ARPA (default: {DEFAULT_LM_PATH})",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Vynutí syntetické dummy audio i když není zadán WAV.",
    )
    parser.add_argument(
        "--dummy-seconds",
        type=float,
        default=2.5,
        help="Délka dummy audia v sekundách (default: 2.5)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=6.0,
        help="Max délka audia z WAV pro rychlý CPU test (default: 6.0)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Volitelná cesta k výstupnímu JSON. Jinak se vytvoří v results/.",
    )
    return parser.parse_args()


def to_float32(audio):
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = max(abs(info.min), info.max)
        return audio.astype(np.float32) / float(scale)
    return audio.astype(np.float32)


def resample_linear(audio, src_sr, dst_sr):
    if src_sr == dst_sr:
        return audio.astype(np.float32)

    duration = len(audio) / float(src_sr)
    dst_len = int(round(duration * dst_sr))
    if dst_len <= 1:
        return np.zeros((1,), dtype=np.float32)

    x_old = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def load_wav_mono(path, max_seconds):
    sample_rate, data = wavfile.read(path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = to_float32(data)

    if max_seconds is not None and max_seconds > 0:
        max_samples = int(max_seconds * sample_rate)
        data = data[:max_samples]

    if sample_rate != 16000:
        data = resample_linear(data, sample_rate, 16000)
        sample_rate = 16000

    return data, sample_rate


def build_dummy_audio(seconds, sample_rate=16000):
    total = max(1, int(seconds * sample_rate))
    t = np.linspace(0.0, seconds, total, endpoint=False)

    signal = 0.15 * np.sin(2 * np.pi * 220.0 * t)
    signal += 0.08 * np.sin(2 * np.pi * 440.0 * t)
    signal += 0.02 * np.random.default_rng(42).normal(size=total)

    fade = int(0.02 * sample_rate)
    if fade > 0 and total > 2 * fade:
        window = np.ones(total, dtype=np.float32)
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        window[:fade] = ramp
        window[-fade:] = ramp[::-1]
        signal = signal * window

    return signal.astype(np.float32), sample_rate


def build_decoder_if_possible(processor, lm_path):
    lm_file = Path(lm_path)
    if not lm_file.exists():
        return None, f"LM soubor neexistuje: {lm_file}"

    try:
        pyctcdecode = importlib.import_module("pyctcdecode")
        build_ctcdecoder = pyctcdecode.build_ctcdecoder
    except Exception as exc:
        return None, f"pyctcdecode není dostupné: {exc}"

    vocab = processor.tokenizer.get_vocab()
    labels = [token for token, _ in sorted(vocab.items(), key=lambda item: item[1])]

    try:
        decoder = build_ctcdecoder(labels=labels, kenlm_model_path=str(lm_file))
        return decoder, f"LM decoder aktivní: {lm_file}"
    except Exception as exc:
        return None, f"LM decoder nelze vytvořit: {exc}"


def decode_greedy(processor, logits):
    pred_ids = torch.argmax(logits, dim=-1)
    texts = processor.batch_decode(pred_ids)
    return (texts[0] if texts else "").strip()


def decode_with_lm(decoder, logits):
    if decoder is None:
        return None

    logits_np = logits.detach().cpu().numpy()
    if logits_np.ndim == 3:
        logits_np = logits_np[0]
    text = decoder.decode(logits_np)
    return (text or "").strip()


def default_output_path(audio_path, used_dummy):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = "dummy" if used_dummy else Path(audio_path).stem
    return PROJECT_ROOT / "results" / f"w2_lm_compare_{stem}_{stamp}.json"


def main():
    args = parse_args()

    run_started = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    model_dir = Path(args.model_dir).expanduser().resolve()
    lm_path = Path(args.lm_path).expanduser().resolve() if args.lm_path else None

    if not model_dir.exists():
        raise SystemExit(f"Model složka neexistuje: {model_dir}")

    use_dummy = args.dummy or not args.audio_path
    if use_dummy:
        audio, sample_rate = build_dummy_audio(seconds=args.dummy_seconds, sample_rate=16000)
        audio_source = "dummy"
    else:
        wav_path = Path(args.audio_path).expanduser().resolve()
        if not wav_path.exists():
            raise SystemExit(f"WAV soubor neexistuje: {wav_path}")
        audio, sample_rate = load_wav_mono(str(wav_path), max_seconds=args.max_seconds)
        audio_source = str(wav_path)

    print(f"Audio source: {audio_source}")
    print(f"Samples: {len(audio)} | sample_rate: {sample_rate}")

    print(f"Načítám model z: {model_dir}")
    processor = Wav2Vec2Processor.from_pretrained(str(model_dir), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(model_dir), local_files_only=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    greedy_text = decode_greedy(processor, logits)

    decoder = None
    lm_message = "LM vypnuto"
    lm_text = None
    if lm_path is not None:
        decoder, lm_message = build_decoder_if_possible(processor, lm_path)
        lm_text = decode_with_lm(decoder, logits)

    runtime = round(time.perf_counter() - t0, 3)

    output_path = Path(args.output).expanduser().resolve() if args.output else default_output_path(args.audio_path, use_dummy)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "model_dir": str(model_dir),
            "lm_path": str(lm_path) if lm_path is not None else None,
            "lm_message": lm_message,
            "audio_source": audio_source,
            "used_dummy_audio": use_dummy,
            "dummy_seconds": args.dummy_seconds if use_dummy else None,
            "max_seconds": args.max_seconds,
            "device": device,
            "logits_shape": list(logits.shape),
            "runtime_seconds": runtime,
            "run_started_utc": run_started.isoformat(),
            "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        },
        "texts": {
            "greedy": greedy_text,
            "lm_decode": lm_text,
        },
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("W2V2 decode compare")
    print(f"Device: {device}")
    print(f"LM status: {lm_message}")
    print(f"Greedy: {greedy_text[:160]}")
    print(f"LM:     {(lm_text or '')[:160]}")
    print(f"Output: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    #main()
    inspect_lm_file(DEFAULT_LM_PATH)
