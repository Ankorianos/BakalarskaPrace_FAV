import numpy as np
from scipy.io import wavfile
import os

def process_audio(file_path):
    """
    Rozdělí stereo WAV na L, R a MONO mix.
    """
    print(f"Načítám soubor: {file_path}")
    sample_rate, data = wavfile.read(file_path)
    
    # Ověření, že jde o stereo
    if len(data.shape) < 2 or data.shape[1] != 2:
        print("Chyba: Soubor není stereo!")
        return
    
    # 1. Rozdělení kanálů
    left_channel = data[:, 0]
    right_channel = data[:, 1]
    
    # 2. Vytvoření MONO mixu (průměr obou kanálů)
    # Musíme dávat pozor na přetečení (overflow) u int16, tak to nejdřív převedeme na float
    mono_mix = ((left_channel.astype(float) + right_channel.astype(float)) / 2).astype(data.dtype)
    
    # Názvy souborů
    base_name = os.path.splitext(file_path)[0]
    l_file = f"{base_name}_L.wav"
    r_file = f"{base_name}_R.wav"
    mono_file = f"{base_name}_MONO.wav"
    
    # 3. Uložení
    print(f"Ukládám Levý kanál -> {l_file}")
    wavfile.write(l_file, sample_rate, left_channel)
    
    print(f"Ukládám Pravý kanál -> {r_file}")
    wavfile.write(r_file, sample_rate, right_channel)
    
    print(f"Ukládám MONO mix -> {mono_file}")
    wavfile.write(mono_file, sample_rate, mono_mix)
    
    print("\nHotovo! Audio data jsou připravena.")

if __name__ == "__main__":
    audio_file = "12008_001.wav"
    if os.path.exists(audio_file):
        process_audio(audio_file)
    else:
        print(f"Soubor {audio_file} nebyl nalezen.")
