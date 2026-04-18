# Hlavní dokumentace ASR projektu

## 1) Přehled pipeline

Projekt obsahuje dvě hlavní ASR větve:

- **MIX**: přepis jedné smíchané mono stopy (`*_MIX.wav`).
- **INDIVIDUAL_SPLIT**: přepis oddělených kanálů mluvčích (`L/R`) a jejich sloučení.

Použité backendy:

- **Whisper** (`backend=whisper`),
- **Interspeech W2V2 + volitelný KenLM** (`backend=interspeech`, `LM.arpa`).

Evaluace pracuje s referencemi v `results/`:

- `ground_truth_eval.json` + `formal_ground_truth_eval.json` (text-level),
- `ground_truth_speakers.json` + `formal_ground_truth_speakers.json` (speaker-level).

---

## 2) Aktuální skripty (`scripts/`)

### 2.1 Audio preprocessing

- `audio_processor.py`
	- načte stereo WAV,
	- provede DC offset removal,
	- aplikuje shared LUFS normalizaci (`TARGET_LUFS = -23`) + shared peak guard,
	- uloží `*_L.wav`, `*_R.wav`, `*_MIX.wav`,
	- vygeneruje `results/*_signal_report.txt`.

- `audio_cleaner_spectral.py`
	- spektrální crosstalk suppression přes STFT/ISTFT,
	- binární maskování podle širokopásmové energie L/R,
	- časové vyhlazení masek,
	- výstup: `*_INDIVIDUAL_spectral.wav`.

### 2.2 ASR inference

- `asr_mix_whisper.py`
	- MIX přepis přes Whisper + Auditok VAD,
	- obsahuje dedup/overlap/adjacent merge postprocessing,
	- výstup: `results/<name>_<scope>_whisper.json`.

- `asr_mix_interspeech.py`
	- MIX přepis přes lokální W2V2 model (`INTERSPEECH2023`),
	- volitelný LM decode (`pyctcdecode` + `LM.arpa`), fallback na greedy decode,
	- VAD chunking + chunk backshift + boundary overlap trim,
	- výstup: `results/<name>_<scope>_interspeech.json`.

- `asr_individual_whisper.py`
	- stereo split (L/R), separátní Whisper přepis obou kanálů,
	- merge + cross-channel dedup segmentů,
	- výstup obsahuje `segments`, `Speaker_L_full_transcription`, `speaker_R_full_transcription`, `full_transcription`.

- `asr_individual_interspeech.py`
	- stereo split (L/R), W2V2 + volitelný LM,
	- VAD chunking + backshift + boundary overlap trim,
	- cross-channel dedup v individual režimu,
	- výstupové schéma odpovídá individual Whisper variantě.

- `w2.py`
	- diagnostický test načtení lokálního W2V2 modelu,
	- kontrola modelových souborů,
	- volitelný LM decoder test a test decode z `Test_sentence.wav`.

### 2.3 Ground truth a evaluace

- `gt_clean_parser.py`
	- převod TRS do `ground_truth_eval.json` a `ground_truth_speakers.json`,
	- respektuje `Turn/Who/Sync` strukturu,
	- generuje segmenty včetně overlap informace.

- `formal_clean_parser.py`
	- převod `FormalForEvaluation.txt` do `formal_ground_truth_eval.json` a `formal_ground_truth_speakers.json`,
	- podporuje fallback encoding (`utf-8`, `cp1250`, ...).

- `evaluate_wer.py`
	- text-level evaluace (`WER`, `SUB/INS/DEL`, top chyby),
	- běží proti `ground_truth_eval.json` i `formal_ground_truth_eval.json`,
	- podporuje WordChanges mapu (`data/WordChanges.txt`) jak pro tab, tak pro 2-token mezery.

- `evaluate_wer_speakers.py`
	- speaker-level evaluace pro individual JSON výstupy,
	- přiřazení ASR kanálů na role interviewer/interviewee dle délky + swap check,
	- stejná podpora WordChanges mapy jako text-level evaluátor.

- `encoding.py`
	- pomocný skript pro detekci encodingu souborů.

---

## 3) Stav `results/` (aktuální snapshot)

V `results/` je aktuálně **17 souborů**:

- **ASR JSON výstupy (6x)**
	- `12008_001_MIX_full_whisper.json` (181 segmentů)
	- `12008_001_MIX_full_interspeech.json` (31 segmentů)
	- `12008_001_L_full_whisper.json` (173 segmentů)
	- `12008_001_R_full_whisper.json` (182 segmentů)
	- `12008_001_INDIVIDUAL_spectral_full_whisper.json` (172 segmentů)
	- `12008_001_INDIVIDUAL_spectral_full_interspeech.json` (52 segmentů)

- **Reference JSON (4x)**
	- `ground_truth_eval.json`
	- `ground_truth_speakers.json`
	- `formal_ground_truth_eval.json`
	- `formal_ground_truth_speakers.json`

- **Eval reporty (6x)**
	- `eval_report_12008_001_MIX_full_whisper.txt`
	- `eval_report_12008_001_MIX_full_interspeech.txt`
	- `eval_report_12008_001_L_full_whisper.txt`
	- `eval_report_12008_001_R_full_whisper.txt`
	- `eval_report_speakers_12008_001_INDIVIDUAL_spectral_full_whisper.txt`
	- `eval_report_speakers_12008_001_INDIVIDUAL_spectral_full_interspeech.txt`

- **Signal report (1x)**
	- `12008_001_signal_report.txt`

Poznámka: uložené JSON odráží konfiguraci v době konkrétního běhu (např. starší Whisper běhy mají `model=large`).

---

## 4) Snapshot metrik z reportů

### 4.1 Text-level (`evaluate_wer.py`)

- `MIX_full_interspeech`
	- vs `ground_truth_eval.json`: **11.86 %**
	- vs `formal_ground_truth_eval.json`: **28.95 %**

- `MIX_full_whisper`
	- vs `ground_truth_eval.json`: **17.03 %**
	- vs `formal_ground_truth_eval.json`: **32.80 %**

- `L_full_whisper`
	- vs `ground_truth_eval.json`: **17.21 %**
	- vs `formal_ground_truth_eval.json`: **32.26 %**

- `R_full_whisper`
	- vs `ground_truth_eval.json`: **18.52 %**
	- vs `formal_ground_truth_eval.json`: **35.12 %**

### 4.2 Speaker-level (`evaluate_wer_speakers.py`)

- `INDIVIDUAL_spectral_full_interspeech`
	- interviewer: **21.72 %**
	- interviewee: **8.57 %**
	- macro: **15.14 %**
	- weighted: **10.52 %**

- `INDIVIDUAL_spectral_full_whisper`
	- interviewer: **27.21 %**
	- interviewee: **12.43 %**
	- macro: **19.82 %**
	- weighted: **14.62 %**

V tomto snapshotu vychází interspeech lépe než whisper v MIX i INDIVIDUAL reportech.

---

## 5) Praktické poznámky

- `evaluate_wer.py` zapisuje obě GT sekce (ground truth + formal) do jednoho `eval_report_*.txt`.
- WordChanges se aplikují na alignment typu `substitute`; report obsahuje i `WordChanges impact (WER)`.
- Speaker evaluátor přiřazuje role mezi dvěma ASR full transkripcemi a uvádí confidence oproti swapped variantě.
- Po každé změně TRS nebo formal vstupu je potřeba znovu spustit parsery (`gt_clean_parser.py`, `formal_clean_parser.py`) před evaluací.

---

## 6) Doporučené pořadí běhu

1. `audio_processor.py` (volitelně `audio_cleaner_spectral.py`).
2. ASR skript dle experimentu (`asr_mix_*` nebo `asr_individual_*`).
3. `evaluate_wer.py` pro text-level srovnání.
4. `evaluate_wer_speakers.py` pro speaker-level srovnání.
