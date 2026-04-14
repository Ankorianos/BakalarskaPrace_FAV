import numpy as np
import sys
from pathlib import Path
from pydub import AudioSegment
from scipy.signal import stft, istft

def suppress_crosstalk_spectral(input_path, output_path, alpha=1.5, power=2.0, window_size=2048):
    """
    Odstraní přeslechy pomocí frekvenčního maskování (STFT).
    
    :param alpha: Síla potlačení. Pro zachycení krátkých slov (např. "ano") volíme nižší (1.1).
    :param power: Nyní se nevyužívá.
    :param window_size: Velikost STFT okna. Menší okno (2048) lépe chytá krátká slova.
    """
    print(f"Načítám {input_path}...")
    audio = AudioSegment.from_file(input_path)
    sr = audio.frame_rate
    
    if audio.channels != 2:
        raise ValueError("Tento skript vyžaduje stereo (2-kanálový) audio soubor.")

    # Převedeme na float32 v rozsahu -1.0 až 1.0 pro matematické operace
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    samples = samples.reshape((-1, 2))
    
    print("Provádím Fourierovu transformaci (STFT) - to chvíli potrvá...")
    # Rozklad do frekvenční domény
    f, t, Zxx_L = stft(samples[:, 0], fs=sr, nperseg=window_size)
    f, t, Zxx_R = stft(samples[:, 1], fs=sr, nperseg=window_size)
    
    # Výpočet magnitud (hlasitosti jednotlivých frekvencí)
    mag_L = np.abs(Zxx_L)
    mag_R = np.abs(Zxx_R)
    
    print("Počítám spektrální masky (širokopásmové binární potlačení pro odstranění plechových zvuků)...")
    
    # Místo odřezávání jednotlivých frekvencí (což vytváří "mimozemské" zvuky - tzv. musical noise),
    # sečteme umocněnou energii pro celý časový okamžik (umocnění víc zvýrazní dominantního řečníka).
    energy_L = np.sum(mag_L, axis=0)
    energy_R = np.sum(mag_R, axis=0)
    
    # Buď hraje celý rámec, nebo je úplné ticho.
    raw_mask_L = np.where(energy_L > (alpha * energy_R), 1.0, 0.0)
    raw_mask_R = np.where(energy_R > (alpha * energy_L), 1.0, 0.0)
    
    print("Vyhlazuji masky v čase pro přirozenější náběhy...")
    # Aplikujeme měkké časové vyhlazení, aby zvuk prudce "neustřihl" začátky a konce slov ("ano")
    # Přes průměr 5 okének vytvoříme velmi krátký fade (aby v těsných přesazích neprosákl druhý hlas).
    smooth_window = np.ones(5) / 5.0
    mask_L = np.convolve(raw_mask_L, smooth_window, mode='same')[np.newaxis, :]
    mask_R = np.convolve(raw_mask_R, smooth_window, mode='same')[np.newaxis, :]
    
    print("Aplikuji masky a rekonstruuji zvuk (ISTFT)...")
    # Vynásobení původních komplexních čísel naší maskou
    _, clean_L = istft(Zxx_L * mask_L, fs=sr)
    _, clean_R = istft(Zxx_R * mask_R, fs=sr)
    
    # Zarovnání délek (ISTFT může vrátit o pár vzorků delší/kratší pole kvůli oknům)
    min_len = min(len(clean_L), len(clean_R), len(samples))
    clean_L = clean_L[:min_len]
    clean_R = clean_R[:min_len]
    
    # Složení zpět do sterea
    processed_samples = np.column_stack((clean_L, clean_R))
    
    # Zamezení clippingu a převod zpět na 16-bit PCM
    np.clip(processed_samples, -1.0, 1.0, out=processed_samples)
    processed_samples = np.round(processed_samples * 32767.0).astype(np.int16)
    
    print(f"Exportuji do {output_path}...")
    processed_audio = audio._spawn(processed_samples.tobytes())
    processed_audio.export(output_path, format="wav")
    print("Export dokončen!")

def main():
    if len(sys.argv) != 2 or not sys.argv[1].strip():
        print("Použití: python audio_cleaner_spectral.py <cesta_k_stereo_wav>")
        sys.exit(1)

    input_path = Path(sys.argv[1].strip()).expanduser().resolve()
    if not input_path.exists():
        print(f"Chyba: Soubor nebyl nalezen: {input_path}")
        sys.exit(1)

    output_path = input_path.with_name(f"{input_path.stem}_INDIVIDUAL_spectral.wav")

    # Pokud nahrávka po zpracování zní moc "pod vodou" nebo plechově, sniž `alpha` (např. na 1.0 nebo 0.8)
    # Pokud tam pořád slyšíš moc přeslechu, zvyš `alpha` (např. na 2.0 nebo 3.0)
    suppress_crosstalk_spectral(
        input_path=str(input_path),
        output_path=str(output_path),
        alpha=1.75,       
        power=2.0,       
        window_size=2048 
    )

    print(f"Hotovo: {output_path}")

if __name__ == "__main__":
    main()