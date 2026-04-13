import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile


EPS = 1e-12


OUTPUT_DIR = "data"
# Výstupní složka pro *_L_gate.wav a *_R_gate.wav.

FRAME_MS = 25.0
# Délka rámce v ms pro rozhodování dominance.

HOP_MS = 10.0
# Posun mezi rámci v ms.

DOMINANCE_DB_CUT = 2
# Pokud je převaha cílového kanálu pod touto hodnotou (dB), rámec se ztlumí na ticho.

AUTO_DOMINANCE_DB_CUT = True
# Pokud True, dominance_db_cut se odhadne automaticky podle nahrávky.

AUTO_DOMINANCE_PERCENTILE = 55
# Percentil z |dominance_db| použitý pro automatický odhad prahu.

AUTO_DOMINANCE_MIN_DB = 1
# Dolní mez automatického prahu dominance.

AUTO_DOMINANCE_MAX_DB = 5
# Horní mez automatického prahu dominance.

NUM_PASSES = 2
# Počet průchodů hard-gate. 3 průchody bývají účinnější na přeslech.

MIN_KEEP_GAIN = 0.07
# Jemný podklad v nedominantních rámcích (0.0 = tvrdé ticho, vyšší = méně záseků).


def to_float32(audio):
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = float(max(abs(info.min), info.max))
        return audio.astype(np.float32) / scale
    return audio.astype(np.float32)


def from_float32(audio_float, target_dtype):
    clipped = np.clip(audio_float, -1.0, 1.0)
    if np.issubdtype(target_dtype, np.integer):
        info = np.iinfo(target_dtype)
        return np.round(clipped * float(info.max)).astype(target_dtype)
    return clipped.astype(target_dtype)


def frame_signal(signal, frame_size, hop_size):
    if len(signal) < frame_size:
        signal = np.pad(signal, (0, frame_size - len(signal)))

    num_frames = 1 + int(np.ceil((len(signal) - frame_size) / hop_size))
    total_len = (num_frames - 1) * hop_size + frame_size
    padded = np.pad(signal, (0, max(0, total_len - len(signal))))

    frames = np.zeros((num_frames, frame_size), dtype=np.float32)
    for idx in range(num_frames):
        start = idx * hop_size
        frames[idx] = padded[start : start + frame_size]

    return frames, len(signal)


def overlap_add(frames, original_len, frame_size, hop_size):
    total_len = (len(frames) - 1) * hop_size + frame_size
    output = np.zeros(total_len, dtype=np.float32)
    weight = np.zeros(total_len, dtype=np.float32)

    window = np.hanning(frame_size).astype(np.float32)
    window = np.maximum(window, 1e-4)

    for idx, frame in enumerate(frames):
        start = idx * hop_size
        output[start : start + frame_size] += frame * window
        weight[start : start + frame_size] += window

    output /= np.maximum(weight, EPS)
    return output[:original_len]


def hard_dominance_gate(target, other, sample_rate, frame_ms=25.0, hop_ms=10.0, dominance_db_cut=2.0):
    frame_size = max(64, int(round(sample_rate * (frame_ms / 1000.0))))
    hop_size = max(32, int(round(sample_rate * (hop_ms / 1000.0))))

    target_frames, original_len = frame_signal(target, frame_size, hop_size)
    other_frames, _ = frame_signal(other, frame_size, hop_size)

    gated_frames = np.zeros_like(target_frames)

    for idx in range(len(target_frames)):
        x = target_frames[idx]
        y = other_frames[idx]

        ex = float(np.mean(x * x))
        ey = float(np.mean(y * y))
        dominance_db = 10.0 * np.log10((ex + EPS) / (ey + EPS))

        if dominance_db >= dominance_db_cut:
            gated_frames[idx] = x
        else:
            gated_frames[idx] = x * MIN_KEEP_GAIN

    return overlap_add(gated_frames, original_len, frame_size, hop_size)


def estimate_dominance_db_cut(left, right, sample_rate, frame_ms=25.0, hop_ms=10.0):
    frame_size = max(64, int(round(sample_rate * (frame_ms / 1000.0))))
    hop_size = max(32, int(round(sample_rate * (hop_ms / 1000.0))))

    left_frames, _ = frame_signal(left, frame_size, hop_size)
    right_frames, _ = frame_signal(right, frame_size, hop_size)

    frame_energy = np.maximum(
        np.mean(left_frames * left_frames, axis=1),
        np.mean(right_frames * right_frames, axis=1),
    )

    if len(frame_energy) == 0:
        return DOMINANCE_DB_CUT

    energy_floor = float(np.percentile(frame_energy, 20))
    mask = frame_energy > max(energy_floor, EPS)

    if not np.any(mask):
        return DOMINANCE_DB_CUT

    ex = np.mean(left_frames[mask] * left_frames[mask], axis=1)
    ey = np.mean(right_frames[mask] * right_frames[mask], axis=1)
    dominance_db_abs = np.abs(10.0 * np.log10((ex + EPS) / (ey + EPS)))

    if len(dominance_db_abs) == 0:
        return DOMINANCE_DB_CUT

    estimated = float(np.percentile(dominance_db_abs, AUTO_DOMINANCE_PERCENTILE))
    estimated = float(np.clip(estimated, AUTO_DOMINANCE_MIN_DB, AUTO_DOMINANCE_MAX_DB))
    return estimated


def peak_normalize_pair(left, right, peak=0.98):
    max_peak = float(max(np.max(np.abs(left)), np.max(np.abs(right)), EPS))
    gain = min(1.0, peak / max_peak)
    return left * gain, right * gain, gain


def build_report(input_path, sample_rate, left, right, left_clean, right_clean, gain, used_dominance_db_cut):
    def rms_db(sig):
        rms = float(np.sqrt(np.mean(sig * sig)))
        return float(20.0 * np.log10(max(rms, EPS)))

    return {
        "input": str(input_path),
        "sample_rate": int(sample_rate),
        "samples": int(len(left)),
        "duration_sec": float(len(left) / sample_rate),
        "params": {
            "frame_ms": FRAME_MS,
            "hop_ms": HOP_MS,
            "dominance_db_cut": used_dominance_db_cut,
            "dominance_db_cut_auto": AUTO_DOMINANCE_DB_CUT,
            "num_passes": NUM_PASSES,
            "min_keep_gain": MIN_KEEP_GAIN,
        },
        "stats": {
            "left_rms_before_db": rms_db(left),
            "left_rms_after_db": rms_db(left_clean),
            "right_rms_before_db": rms_db(right),
            "right_rms_after_db": rms_db(right_clean),
            "corr_before": float(np.corrcoef(left, right)[0, 1]) if np.std(left) > 0 and np.std(right) > 0 else 0.0,
            "corr_after": float(np.corrcoef(left_clean, right_clean)[0, 1])
            if np.std(left_clean) > 0 and np.std(right_clean) > 0
            else 0.0,
            "shared_peak_gain": float(gain),
        },
    }


def process_file(input_wav, output_dir):
    sample_rate, data = wavfile.read(input_wav)

    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("Vstup musí být stereo WAV (2 kanály).")

    original_dtype = data.dtype
    left = to_float32(data[:, 0])
    right = to_float32(data[:, 1])

    if AUTO_DOMINANCE_DB_CUT:
        dominance_db_cut_used = estimate_dominance_db_cut(
            left,
            right,
            sample_rate,
            frame_ms=FRAME_MS,
            hop_ms=HOP_MS,
        )
    else:
        dominance_db_cut_used = float(DOMINANCE_DB_CUT)

    left_clean = left.copy()
    right_clean = right.copy()

    for _ in range(max(1, int(NUM_PASSES))):
        left_clean = hard_dominance_gate(
            target=left_clean,
            other=right,
            sample_rate=sample_rate,
            frame_ms=FRAME_MS,
            hop_ms=HOP_MS,
            dominance_db_cut=dominance_db_cut_used,
        )
        right_clean = hard_dominance_gate(
            target=right_clean,
            other=left,
            sample_rate=sample_rate,
            frame_ms=FRAME_MS,
            hop_ms=HOP_MS,
            dominance_db_cut=dominance_db_cut_used,
        )

    left_clean, right_clean, shared_gain = peak_normalize_pair(left_clean, right_clean)

    stem = Path(input_wav).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path("results")
    report_dir.mkdir(parents=True, exist_ok=True)

    left_path = output_dir / f"{stem}_L_gate.wav"
    right_path = output_dir / f"{stem}_R_gate.wav"
    report_path = report_dir / f"{stem}_gate_report.json"

    wavfile.write(left_path, sample_rate, from_float32(left_clean, original_dtype))
    wavfile.write(right_path, sample_rate, from_float32(right_clean, original_dtype))

    report = build_report(
        input_path=input_wav,
        sample_rate=sample_rate,
        left=left,
        right=right,
        left_clean=left_clean,
        right_clean=right_clean,
        gain=shared_gain,
        used_dominance_db_cut=dominance_db_cut_used,
    )

    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    return left_path, right_path, report_path, report


def main():
    if len(sys.argv) < 2:
        print("Použití: python scripts/audio_cleaner_gate.py data/12008_001.wav")
        return

    input_wav = Path(sys.argv[1])
    output_dir = Path(OUTPUT_DIR)

    if not input_wav.exists():
        raise FileNotFoundError(f"Soubor nebyl nalezen: {input_wav}")

    left_path, right_path, report_path, report = process_file(
        input_wav=input_wav,
        output_dir=output_dir,
    )

    print("Hotovo: hard-gate potlačení přeslechu dokončeno.")
    print(f"Vstup: {input_wav}")
    print(f"Výstup L: {left_path}")
    print(f"Výstup R: {right_path}")
    print(f"Report: {report_path}")
    print(
        "Shrnutí: corr_before={:.4f}, corr_after={:.4f}, shared_peak_gain={:.6f}".format(
            report["stats"]["corr_before"],
            report["stats"]["corr_after"],
            report["stats"]["shared_peak_gain"],
        )
    )


if __name__ == "__main__":
    main()
