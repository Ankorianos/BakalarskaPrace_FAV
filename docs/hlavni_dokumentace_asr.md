# Hlavní dokumentace ASR projektu

## 1) Aktuální směr

Projekt je rozdělen na dvě jasné větve:

- MIX: referenční baseline bez separace,
- INDIVIDUAL: hlavní experimentální větev se separací.

Cíl je obhajitelně ukázat, že INDIVIDUAL větev po separaci zlepší přepis proti MIX baseline.

---

## 2) Aktuální skripty (scripts)

- `scripts/audio_processor.py` -> příprava L/R + MIX audia
- `scripts/asr_mix_whisper.py` -> ASR pro MIX větev
- `scripts/asr_individual_whisper.py` -> ASR pro INDIVIDUAL baseline (bez separace)
- `scripts/evaluate_wer.py` -> jednotné text-level vyhodnocení

Poznámka: větev fastwhisper byla odstraněna; aktivně se drží klasický OpenAI Whisper.

---

## 3) Výstupní soubory

Typické výstupy:

- `results/mix_results_range_whisper.json`
- `results/12008_001_range_whisper.json` (resp. `<recording>_range_whisper.json`)
- `results/eval_report_*.txt`

INDIVIDUAL výstup obsahuje navíc:

- `Speaker_L_full_transcription`
- `speaker_R_full_transcription`

Všechny varianty mají být vyhodnocovány stejným evaluátorem a stejným eval rozsahem.

---

## 4) Eval protokol (zamknout)

Pro férové porovnání držet:

- stejné referenční GT,
- stejné časové okno,
- stejné normalizační kroky,
- stejný `scripts/evaluate_wer.py`.

---

## 5) Co je hotové

- stabilní MIX baseline,
- stabilní INDIVIDUAL baseline bez separace,
- jednotný evaluator,
- sjednocené názvosloví ve skriptech a výstupech.

---

## 6) Co je další hlavní krok

1. Přidat separační variantu do INDIVIDUAL větve.
2. Vygenerovat výsledky na stejném eval rozsahu jako MIX.
3. Vyhodnotit přes stejný evaluator.
4. Zapsat tabulkové porovnání MIX vs INDIVIDUAL.

---

## 6.1) Speaker-level evaluace (praktické pravidlo)

- V INDIVIDUAL větvi lze dělat speaker-level evaluaci i bez diarizace, protože po separaci má každá stopa odpovídat jednomu mluvčímu.
- V MIX větvi speaker-level evaluaci bez spolehlivé diarizace nedělat, protože text není čistě přiřaditelný jednomu mluvčímu.
- Pro MIX větev proto držet text-level WER; speaker-level až pokud bude validní diarizační krok.

---

## 7) Co teď není hlavní fokus

- rozšiřování experimentů mimo MIX/INDIVIDUAL osu,
- další refaktor baseline bez přímého přínosu k porovnání,
- komplikování TODO o vedlejší směry.

---

## 8) Shrnutí jednou větou

MIX je referenční baseline, INDIVIDUAL je hlavní větev se separací, a celý projekt je veden tak, aby šlo čistě a měřitelně porovnat přínos separace na ASR.
