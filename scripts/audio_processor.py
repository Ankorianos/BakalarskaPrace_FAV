import numpy as np
from scipy.io import wavfile
import os
from pathlib import Path

try:
    import pyloudnorm as pyln
except Exception:
    pyln = None


TARGET_PEAK = 0.95
TARGET_LUFS = -23.0


def to_float32(audio):
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = max(abs(info.min), info.max)
        return audio.astype(np.float32) / float(scale)
    return audio.astype(np.float32)


def from_float32(audio_float, target_dtype):
    clipped = np.clip(audio_float, -1.0, 1.0)
    if np.issubdtype(target_dtype, np.integer):
        info = np.iinfo(target_dtype)
        scaled = clipped * float(info.max)
        return np.round(scaled).astype(target_dtype)
    return clipped.astype(target_dtype)


def remove_dc_offset(signal):
    return signal - np.mean(signal)


def apply_shared_peak_normalization(left, right, target_peak=TARGET_PEAK):
    global_peak = max(np.max(np.abs(left)), np.max(np.abs(right)))
    if global_peak <= 1e-9:
        return left, right, 1.0

    gain = min(1.0, target_peak / global_peak)
    return left * gain, right * gain, gain


def apply_shared_lufs_normalization(left, right, sample_rate, target_lufs=TARGET_LUFS):
    if pyln is None:
        return left, right, 1.0, None

    stereo = np.column_stack((left, right))

    try:
        meter = pyln.Meter(sample_rate)
        measured_lufs = meter.integrated_loudness(stereo)
    except Exception:
        return left, right, 1.0, None

    gain_db = target_lufs - measured_lufs
    gain_linear = float(10.0 ** (gain_db / 20.0))

    left_norm = left * gain_linear
    right_norm = right * gain_linear

    # Společná ochrana proti clippingu (zachová stereo poměr)
    peak_after = max(np.max(np.abs(left_norm)), np.max(np.abs(right_norm)))
    clip_guard_gain = 1.0 if peak_after <= 1.0 else (1.0 / peak_after)

    total_gain = gain_linear * clip_guard_gain
    return left * total_gain, right * total_gain, total_gain, measured_lufs


def _to_db(value, eps=1e-12):
    return float(20.0 * np.log10(max(float(value), eps)))


def _rms(signal):
    return float(np.sqrt(np.mean(np.square(signal))))


def _peak(signal):
    return float(np.max(np.abs(signal)))


def _safe_correlation(left, right):
    left_std = np.std(left)
    right_std = np.std(right)
    if left_std <= 1e-12 or right_std <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _integrated_lufs(signal, sample_rate):
    if pyln is None:
        return None
    try:
        meter = pyln.Meter(sample_rate)
        return float(meter.integrated_loudness(signal))
    except Exception:
        return None


def build_signal_report(
    original_left,
    original_right,
    processed_left,
    processed_right,
    sample_rate,
    source_file,
    lufs_gain,
    peak_gain,
):
    original_mid = 0.5 * (original_left + original_right)
    original_side = 0.5 * (original_left - original_right)

    processed_mid = 0.5 * (processed_left + processed_right)
    processed_side = 0.5 * (processed_left - processed_right)

    orig_mid_rms = _rms(original_mid)
    orig_side_rms = _rms(original_side)
    proc_mid_rms = _rms(processed_mid)
    proc_side_rms = _rms(processed_side)

    report = {
        "source_file": str(source_file),
        "sample_rate": int(sample_rate),
        "num_samples": int(len(processed_left)),
        "duration_sec": float(len(processed_left) / sample_rate),
        "gains": {
            "lufs_gain": float(lufs_gain),
            "peak_gain": float(peak_gain),
            "total_shared_gain": float(lufs_gain * peak_gain),
        },
        "stereo": {
            "correlation_before": _safe_correlation(original_left, original_right),
            "correlation_after": _safe_correlation(processed_left, processed_right),
        },
        "left_channel": {
            "rms_dbfs_before": _to_db(_rms(original_left)),
            "rms_dbfs_after": _to_db(_rms(processed_left)),
            "peak_dbfs_before": _to_db(_peak(original_left)),
            "peak_dbfs_after": _to_db(_peak(processed_left)),
        },
        "right_channel": {
            "rms_dbfs_before": _to_db(_rms(original_right)),
            "rms_dbfs_after": _to_db(_rms(processed_right)),
            "peak_dbfs_before": _to_db(_peak(original_right)),
            "peak_dbfs_after": _to_db(_peak(processed_right)),
        },
        "mid_side": {
            "mid_rms_dbfs_before": _to_db(orig_mid_rms),
            "mid_rms_dbfs_after": _to_db(proc_mid_rms),
            "side_rms_dbfs_before": _to_db(orig_side_rms),
            "side_rms_dbfs_after": _to_db(proc_side_rms),
            "side_to_mid_db_before": _to_db(orig_side_rms / max(orig_mid_rms, 1e-12)),
            "side_to_mid_db_after": _to_db(proc_side_rms / max(proc_mid_rms, 1e-12)),
        },
        "lufs": {
            "stereo_before": _integrated_lufs(np.column_stack((original_left, original_right)), sample_rate),
            "stereo_after": _integrated_lufs(np.column_stack((processed_left, processed_right)), sample_rate),
            "mid_before": _integrated_lufs(original_mid, sample_rate),
            "mid_after": _integrated_lufs(processed_mid, sample_rate),
            "side_before": _integrated_lufs(original_side, sample_rate),
            "side_after": _integrated_lufs(processed_side, sample_rate),
            "target_lufs": float(TARGET_LUFS),
        },
    }

    return report


def save_signal_report(report, source_file):
    workspace_root = Path(__file__).resolve().parents[1]
    results_dir = workspace_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(source_file).stem
    txt_path = results_dir / f"{stem}_signal_report.txt"

    lines = [
        "Signal Processing Report",
        "========================",
        f"Source: {report['source_file']}",
        f"Sample rate: {report['sample_rate']} Hz",
        f"Duration: {report['duration_sec']:.2f} s",
        "",
        "Gains",
        "-----",
        f"LUFS shared gain: {report['gains']['lufs_gain']:.6f}",
        f"Peak shared gain: {report['gains']['peak_gain']:.6f}",
        f"Total shared gain: {report['gains']['total_shared_gain']:.6f}",
        "",
        "Stereo",
        "------",
        f"Correlation before: {report['stereo']['correlation_before']:.6f}",
        f"Correlation after:  {report['stereo']['correlation_after']:.6f}",
        "",
        "Mid/Side",
        "--------",
        f"Mid RMS before:  {report['mid_side']['mid_rms_dbfs_before']:.2f} dBFS",
        f"Mid RMS after:   {report['mid_side']['mid_rms_dbfs_after']:.2f} dBFS",
        f"Side RMS before: {report['mid_side']['side_rms_dbfs_before']:.2f} dBFS",
        f"Side RMS after:  {report['mid_side']['side_rms_dbfs_after']:.2f} dBFS",
        f"Side/Mid before: {report['mid_side']['side_to_mid_db_before']:.2f} dB",
        f"Side/Mid after:  {report['mid_side']['side_to_mid_db_after']:.2f} dB",
        "",
        "LUFS",
        "----",
        f"Stereo before: {report['lufs']['stereo_before']}",
        f"Stereo after:  {report['lufs']['stereo_after']}",
        f"Mid before:    {report['lufs']['mid_before']}",
        f"Mid after:     {report['lufs']['mid_after']}",
        f"Side before:   {report['lufs']['side_before']}",
        f"Side after:    {report['lufs']['side_after']}",
        f"Target LUFS:   {report['lufs']['target_lufs']}",
        "",
        "Interpretace metrik",
        "-------------------",
        "LUFS: subjektivně vnímaná hlasitost; cíl sjednocuje celkovou úroveň nahrávek.",
        "Shared gain: jeden společný zesilovací koeficient pro L i R (zachová stereo poměr).",
        "Peak dBFS: nejvyšší okamžitá amplituda; blízko 0 dBFS hrozí clipping.",
        "RMS dBFS: průměrná energetická úroveň signálu (stabilnější než peak).",
        "Correlation L/R: podobnost kanálů; +1 téměř stejné, 0 nezávislé, -1 opačná fáze.",
        "Mid = (L+R)/2: společná složka, obvykle 'střed' scény.",
        "Side = (L-R)/2: rozdílová složka, nese stereo rozdíly mezi kanály.",
        "Side/Mid dB: relativní šířka sterea; vyšší hodnota znamená výraznější stereo separaci.",
        "Before/After: porovnání stavu před a po preprocessingu stejného signálu.",
    ]

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return txt_path

def process_audio(file_path):
    """
    Rozdělí stereo WAV na L, R a MIX stopu.
    Používá jemné předzpracování, které zachová stereo poměr:
    - odstranění DC offsetu,
    - volitelná LUFS normalizace společným gainem,
    - společná (shared) peak normalizace obou kanálů jedním gainem.
    """
    print(f"Načítám soubor: {file_path}")
    sample_rate, data = wavfile.read(file_path)
    
    # Ověření, že jde o stereo
    if len(data.shape) < 2 or data.shape[1] != 2:
        print("Chyba: Soubor není stereo!")
        return

    original_dtype = data.dtype

    # 1. Rozdělení + konverze do float
    left_original = to_float32(data[:, 0])
    right_original = to_float32(data[:, 1])
    left_channel = left_original.copy()
    right_channel = right_original.copy()

    # 2. Jemné čištění (bez změny stereo struktury)
    left_channel = remove_dc_offset(left_channel)
    right_channel = remove_dc_offset(right_channel)
    left_channel, right_channel, lufs_gain, measured_lufs = apply_shared_lufs_normalization(
        left_channel,
        right_channel,
        sample_rate,
    )
    left_channel, right_channel, peak_gain = apply_shared_peak_normalization(left_channel, right_channel)

    # 3. MIX stopa jako průměr z již vyčištěných kanálů
    mono_mix = 0.5 * (left_channel + right_channel)

    # 4. Návrat do původního dtype
    left_out = from_float32(left_channel, original_dtype)
    right_out = from_float32(right_channel, original_dtype)
    mono_out = from_float32(mono_mix, original_dtype)
    
    # Názvy souborů
    base_name = os.path.splitext(file_path)[0]
    l_file = f"{base_name}_L.wav"
    r_file = f"{base_name}_R.wav"
    mix_file = f"{base_name}_MIX.wav"

    # 5. Uložení
    print(f"Ukládám Levý kanál -> {l_file}")
    wavfile.write(l_file, sample_rate, left_out)
    
    print(f"Ukládám Pravý kanál -> {r_file}")
    wavfile.write(r_file, sample_rate, right_out)
    
    print(f"Ukládám MIX stopu -> {mix_file}")
    wavfile.write(mix_file, sample_rate, mono_out)

    report = build_signal_report(
        original_left=left_original,
        original_right=right_original,
        processed_left=left_channel,
        processed_right=right_channel,
        sample_rate=sample_rate,
        source_file=file_path,
        lufs_gain=lufs_gain,
        peak_gain=peak_gain,
    )
    report_txt = save_signal_report(report, file_path)

    if measured_lufs is None:
        print("LUFS normalizace: přeskočena (chybí pyloudnorm nebo nešlo spočítat loudness).")
    else:
        print(f"LUFS normalizace: {measured_lufs:.2f} -> cíl {TARGET_LUFS:.2f} LUFS")

    print(
        f"\nHotovo! Audio data jsou připravena. "
        f"Shared gain LUFS: {lufs_gain:.4f}, peak: {peak_gain:.4f}"
    )
    print(f"Signal report TXT  -> {report_txt}")

if __name__ == "__main__":
    audio_file = str(Path(__file__).resolve().parents[1] / "data" / "12008_001.wav")
    if os.path.exists(audio_file):
        process_audio(audio_file)
    else:
        print(f"Soubor {audio_file} nebyl nalezen.")
