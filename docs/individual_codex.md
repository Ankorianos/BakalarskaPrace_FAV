# INDIVIDUAL codex: hlavní větev se separací

## 1) Role INDIVIDUAL větve

INDIVIDUAL je hlavní implementační větev práce.
Cíl je rozdělit vstupní dvoukanálový záznam na dvě čistší individuální stopy a zlepšit ASR proti MIX baseline.

---

## 2) Cílová pipeline (INDIVIDUAL)

1. Vstup: dvoukanálový záznam.
2. Separace: potlačit přeslech mezi mluvčími.
3. ASR: přepis každé individuální stopy.
4. Sloučení: jednotné JSON schema.
5. Vyhodnocení: `scripts/evaluate_wer.py` na stejném rozsahu jako MIX.

---

## 3) Co je hlavní metrika úspěchu

- zlepšení WER proti MIX baseline,
- nižší počet chyb z přeslechu,
- reprodukovatelné výsledky na stejném eval protokolu.

### Speaker-level evaluace v INDIVIDUAL

Po separaci lze dělat speaker-level evaluaci i bez diarizace:

- stopa A = speaker A,
- stopa B = speaker B.

Podmínka je stabilní mapování stop na referenční labely (např. `interviewer` / `interviewee`) v celém běhu.

---

## 4) Co držet stabilní

- neměnit MIX baseline během separačních experimentů,
- neměnit evaluator mezi běhy,
- neměnit eval rozsah mezi variantami.

---

## 5) Doporučená struktura skriptů

- `scripts/asr_individual.py` -> baseline bez separace (L/R zpracování jednotlivě)
- `scripts/asr_individual_separation.py` -> hlavní separační experiment
- `scripts/evaluate_wer.py` -> jednotné vyhodnocení všech variant

---

## 6) TODO list (INDIVIDUAL)

### 6.1 Baseline pro porovnání

- [ ] Vygenerovat `results/individual_results_range.json` přes `scripts/asr_individual.py`.
- [ ] Vyhodnotit `individual_results_range.json` přes `scripts/evaluate_wer.py`.
- [ ] Zapsat baseline metriky (runtime, WER strict, WER robust).

### 6.2 Separační větev

- [ ] Vytvořit `scripts/asr_individual_separation.py` bez zásahu do baseline skriptů.
- [ ] Zvolit první reprodukovatelnou separační konfiguraci.
- [ ] Uložit výstup ve stejném JSON stylu jako baseline.
- [ ] Uložit metadata separace (metoda/model/parametry/verze).

### 6.3 Srovnání MIX vs INDIVIDUAL

- [ ] Připravit jednotnou tabulku: MIX baseline, INDIVIDUAL baseline, INDIVIDUAL + separace.
- [ ] U všech variant držet stejný eval rozsah.
- [ ] U všech variant držet stejné normalizační kroky a stejný evaluator.
- [ ] U INDIVIDUAL doplnit i speaker-level WER (bez diarizace, podle stopy).
- [ ] U MIX ponechat pouze text-level WER.
- [ ] Dopsat krátkou interpretaci: kde separace pomohla a kde ne.

---

## 7) Shrnutí jednou větou

INDIVIDUAL větev je hlavní směr práce: cílem je přes separaci dosáhnout lepšího přepisu než MIX baseline při stejném eval protokolu.
