# INDIVIDUAL dokumentace

## 1) Skript

Aktuální baseline skript je:

- `scripts/asr_individual_whisper.py`

---

## 2) Spuštění přes sys.argv

Skript bere vstup z `sys.argv[1]` (volitelné). Podporované varianty:

- recording ID: `12008_001`
- cesta na `_L.wav`: `data/12008_001_L.wav`
- cesta na `_R.wav`: `data/12008_001_R.wav`
- cesta bez suffixu: `data/12008_001.wav` (skript dopočítá `_L` + `_R`)

Pokud argument nezadáš, použije se default `data/12008_001_L.wav` + `data/12008_001_R.wav`.

Skript navíc řeší i relativní cestu začínající názvem root složky (např. `Main_Workspace/data/...`) bez zdvojení cesty.

---

## 3) Výstup JSON

Ve výstupu jsou klíče:

- `segments`
- `Speaker_L_full_transcription`
- `speaker_R_full_transcription`
- `full_transcription`

`full_transcription` je sloučený text po deduplikaci mezi kanály.

---

## 4) Doporučený eval tok

1. Vygenerovat `results/*_range_whisper.json` přes `scripts/asr_individual_whisper.py`.
2. Vyhodnotit přes `scripts/evaluate_wer.py` na stejném časovém rozsahu jako MIX.
3. Porovnat strict/robust WER proti MIX baseline.
