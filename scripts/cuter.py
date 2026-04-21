import argparse
from pathlib import Path

from scipy.io import wavfile


def parse_args():
    parser = argparse.ArgumentParser(
        description="Vystřihne úsek ze souboru data/12008_001.wav a uloží ho jako Test_cut.wav do rootu projektu."
    )
    parser.add_argument("start", type=float, help="Začátek v sekundách")
    parser.add_argument("end", type=float, help="Konec v sekundách")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.start < 0:
        raise ValueError("Parametr 'start' musí být >= 0.")
    if args.end <= args.start:
        raise ValueError("Parametr 'end' musí být větší než 'start'.")

    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / "12008_001.wav"
    output_path = project_root / "Test_cut.wav"

    if not input_path.exists():
        raise FileNotFoundError(f"Vstupní soubor nebyl nalezen: {input_path}")

    sample_rate, data = wavfile.read(input_path)
    total_samples = len(data)

    start_sample = int(args.start * sample_rate)
    end_sample = int(args.end * sample_rate)

    if start_sample >= total_samples:
        raise ValueError("Začátek střihu je mimo délku nahrávky.")

    end_sample = min(end_sample, total_samples)
    cut_data = data[start_sample:end_sample]

    if len(cut_data) == 0:
        raise ValueError("Výsledný výstřih je prázdný.")

    wavfile.write(output_path, sample_rate, cut_data)
    print(f"Hotovo: {output_path}")


if __name__ == "__main__":
    main()
