# MIX codex: srovnávací větev bez separace

## 1) Role MIX větve

MIX větev je referenční baseline.
Používá se jeden smíchaný kanál (L+R -> MIX) bez separace mluvčích.

Smysl:

- mít stabilní srovnávací bod,
- mít reprodukovatelný text-level WER,
- porovnat ji proti INDIVIDUAL větvi se separací.

---

## 2) Aktuální pipeline (MIX)

1. `scripts/audio_processor.py` připraví `*_MIX.wav`.
2. `scripts/asr_mix_whisper.py` udělá ASR nad MIX signálem.
3. `scripts/evaluate_wer.py` udělá text-level vyhodnocení.

Výstupy:

- `results/mix_results_range_whisper.json` nebo `results/mix_results_full_whisper.json`
- `results/eval_report_mix_results_*.txt`

---

## 3) Co je v MIX větvi cílem

- stabilita a reprodukovatelnost,
- férové porovnání rozsahu, runtime a WER,
- kvalitní referenční tabulka pro finální porovnání.

MIX větev není hlavní místo pro pokročilé experimenty.

---

## 4) Co teď nedělat v MIX větvi

- neměnit často baseline parametry,
- nepřidávat novou experimentální logiku do baseline skriptu,
- nemíchat sem separační experimenty.

---

## 5) Co zapisovat do BP z MIX větve

- použitý skript a parametry běhu,
- runtime,
- WER strict/robust,
- 2–3 typické textové chyby.

---

## 6) TODO list (MIX)

### 6.1 Stabilní baseline

- [x] Připravit MIX audio (`*_MIX.wav`) ze stejného vstupu.
- [x] Spustit ASR přes `scripts/asr_mix_whisper.py`.
- [x] Uložit výsledný JSON do `results/`.
- [x] Vyhodnotit WER přes `scripts/evaluate_wer.py`.

### 6.2 Dokumentace výsledků

- [ ] Doplnit finální tabulku MIX: rozsah, runtime, WER strict, WER robust.
- [ ] Doplnit krátkou textovou interpretaci MIX výsledků.
- [ ] Uložit 2–3 reprezentativní chyby z MIX baseline.

### 6.3 Připrava na finální srovnání

- [ ] Držet stejný eval rozsah jako u INDIVIDUAL větve.
- [ ] Držet stejný evaluator a stejná normalizační pravidla.
- [ ] Připravit řádek „MIX baseline“ do finální souhrnné tabulky.

---

## 7) Shrnutí jednou větou

MIX větev je stabilní referenční baseline bez separace, která slouží pro čisté porovnání proti INDIVIDUAL separační větvi.
