# INDIVIDUAL codex: separace mluvčích před ASR

## 1) Role INDIVIDUAL větve

INDIVIDUAL větev je hlavní experimentální směr:

- každý mluvčí se přepisuje zvlášť (L/R větev),
- minimalizuje se vzájemný přeslech před ASR,
- výsledky se porovnávají proti MIX baseline.

Smysl: doložit, že využití stereo informace a separace zlepší rozpoznávání řeči.

---

## 2) Aktuální pipeline (INDIVIDUAL)

1. `scripts/audio_processor.py` připraví `*_L.wav` a `*_R.wav` (a `*_MIX.wav` pro baseline).
2. Volitelně `scripts/audio_cleaner_spectral.py` vytvoří `*_INDIVIDUAL_spectral.wav`.
3. ASR běží odděleně pro každý kanál přes:
	- `scripts/asr_individual_whisper.py`, nebo
	- `scripts/asr_individual_interspeech.py`.
4. Výsledky se hodnotí:
	- text-level: `scripts/evaluate_wer.py`,
	- speaker-level: `scripts/evaluate_wer_speakers.py`.

Aktuální výstupy v `results/`:

- `12008_001_INDIVIDUAL_spectral_full_whisper.json`
- `12008_001_INDIVIDUAL_spectral_full_interspeech.json`
- `eval_report_speakers_12008_001_INDIVIDUAL_spectral_full_whisper.txt`
- `eval_report_speakers_12008_001_INDIVIDUAL_spectral_full_interspeech.txt`

---

## 3) Výstupní JSON schéma

INDIVIDUAL JSON obsahuje minimálně:

- `segments`
- `Speaker_L_full_transcription`
- `speaker_R_full_transcription`
- `full_transcription`
- `metadata`

`full_transcription` je sloučený text po deduplikaci mezi kanály.

---

## 4) Doporučený eval tok

1. Vygenerovat INDIVIDUAL JSON (`*_full_whisper.json` / `*_full_interspeech.json`).
2. Spustit `scripts/evaluate_wer.py` pro text-level porovnání s MIX baseline.
3. Spustit `scripts/evaluate_wer_speakers.py` pro role interviewer/interviewee.
4. Použít stejný eval rozsah, stejný evaluator a stejná normalizační pravidla jako u MIX.

---

## 5) TODO list (INDIVIDUAL)

### 5.1 Implementace a běh

- [x] Připravit L/R data ze stejného vstupu (`audio_processor.py`).
- [x] Přidat separační variantu přes spectral cleaning (`audio_cleaner_spectral.py`).
- [x] Mít INDIVIDUAL běh pro Whisper (`asr_individual_whisper.py`).
- [x] Mít INDIVIDUAL běh pro Interspeech W2V2+LM (`asr_individual_interspeech.py`).
- [x] Ukládat výsledky do JSON se sjednoceným schématem.

### 5.2 Evaluace

- [x] Vyhodnocovat text-level WER přes `evaluate_wer.py`.
- [x] Vyhodnocovat speaker-level přes `evaluate_wer_speakers.py`.
- [x] Držet stejný eval rozsah vůči MIX baseline.
- [x] Držet stejnou normalizaci/evaluator pravidla jako v MIX.
- [ ] Dopsat runtime tabulku (preprocessing / ASR / evaluace) pro INDIVIDUAL běhy.

### 5.3 Kvalitativní část do BP

- [ ] Přidat 2–3 reprezentativní párové ukázky (MIX vs INDIVIDUAL) se stejným časovým úsekem.
- [ ] Přidat typologii chyb po separaci (co se zlepšilo / co zůstává problém).
- [ ] Explicitně uvést, kde separace nepomohla (fair discussion).

### 5.4 Finální porovnání do BP

- [ ] Připravit finální souhrnnou tabulku MIX vs INDIVIDUAL (Whisper + Interspeech).
- [ ] Uvést relativní zlepšení proti MIX baseline (pokles WER v %).
- [ ] Dopsat krátké „threats to validity“ (jedna nahrávka, doménová závislost, citlivost na parametry VAD/separace).
- [ ] Přidat krátkou ablační sekci: INDIVIDUAL bez spectral cleaningu vs se spectral cleaningem.

---

## 6) Shrnutí jednou větou

INDIVIDUAL větev realizuje separační přístup „jeden mluvčí = jeden ASR vstup“ a tvoří klíčový důkazní směr, že stereo informace může zlepšit ASR oproti MIX baseline.
