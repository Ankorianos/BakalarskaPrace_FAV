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
2. `scripts/asr_mix_whisper/_interspeech.py` udělá ASR nad MIX signálem.
3. `scripts/evaluate_wer.py` udělá text-level vyhodnocení.

Výstupy:

- `results/12008_001_MIX_full_whisper.json`
- `results/12008_001_MIX_full_interspeech.json`
- `results/eval_report_12008_001_MIX_full_whisper.txt`
- `results/eval_report_12008_001_MIX_full_interspeech.txt`

---

## 3) Co je v MIX větvi cílem

- stabilita a reprodukovatelnost,
- férové porovnání rozsahu, runtime a WER,
- kvalitní referenční tabulka pro finální porovnání.

MIX větev není hlavní místo pro pokročilé experimenty.

### Poznámka ke speaker-level evaluaci

V MIX větvi speaker-level evaluaci bez diarizace nedělat.
Důvod: smíchaný signál neumožňuje spolehlivé přiřazení textu k jednomu mluvčímu.
Prakticky tedy držet MIX větev na text-level WER.

---

## 4) Co teď nedělat v MIX větvi

- neměnit často baseline parametry,
- nepřidávat novou experimentální logiku do baseline skriptu,
- nemíchat sem separační experimenty.

---

## 5) Co zapisovat do BP z MIX větve

- použitý skript a parametry běhu,
- runtime,
- 2–3 typické textové chyby.
- WER s nahrazováním i bez

---

## 6) TODO list (MIX)

### 6.1 Stabilní baseline

- [x] Připravit MIX audio (`*_MIX.wav`) ze stejného vstupu.
- [x] Spustit ASR přes `scripts/asr_mix_whisper.py`.
- [x] Uložit výsledný JSON do `results/`.
- [x] Vyhodnotit WER přes `scripts/evaluate_wer.py`.

### 6.2 Dokumentace výsledků

- [ ] Doplnit finální tabulku MIX: rozsah, runtime, WER s nahrazováním i bez
- [x] Doplnit krátkou textovou interpretaci MIX výsledků.
- [ ] Uložit 2–3 reprezentativní chyby z MIX baseline.

### 6.3 Připrava na finální srovnání

- [x] Držet stejný eval rozsah jako u INDIVIDUAL větve.
- [x] Držet stejný evaluator a stejná normalizační pravidla.
- [ ] Připravit řádek „MIX baseline“ do finální souhrnné tabulky.
- [x] U MIX evidovat jen text-level metriky (speaker-level nehodnotit bez diarizace).

---

## 7) Shrnutí jednou větou

MIX větev je stabilní referenční baseline bez separace, která slouží pro čisté porovnání proti INDIVIDUAL separační větvi.

---

## 8) Co přesně znamenají neodškrtnuté TODO

### 8.1 „Doplnit finální tabulku MIX: rozsah, runtime, WER strict, WER robust“

Prakticky to znamená mít v BP jednu jasnou tabulku pro MIX, kde bude pro každý běh:

- název experimentu (např. MIX+Whisper, MIX+Interspeech),
- eval rozsah (full/range, ID nahrávky),
- runtime (ideálně zvlášť preprocessing / ASR / evaluace),
- WER s nahrazováním i bez

Smysl: čtenář musí vidět nejen přesnost, ale i výpočetní cenu baseline.

### 8.2 „Uložit 2–3 reprezentativní chyby z MIX baseline“

Nejde o náhodné chyby, ale o typické vzory:

- záměna krátkých funkčních slov,
- slévání řeči při překryvu mluvčích,
- chyby u vlastních jmen/čísel.

Smysl: kvalitativně ukázat, kde MIX baseline selhává a proč může stereo pomoci.

### 8.3 „Připravit řádek MIX baseline do finální souhrnné tabulky“

V závěrečné srovnávací tabulce musí být samostatný řádek pro MIX jako referenční bod proti INDIVIDUAL variantám.

Smysl: bez tohoto řádku se hůř obhajuje tvrzení, že stereo zpracování je přínosné.

---

## 9) Vazba na zadání BP

Ano, tato část je v přímé vazbě na zadání BP:

- „Analyzujte možnosti využití stereo…“ -> porovnání MIX vs INDIVIDUAL tuto část přímo pokrývá.
- „Implementujte vybrané metody“ -> skripty v `scripts/` to naplňují.
- „Vyhodnoťte výsledky… a diskutujte přínosy“ -> evaluace a tabulkové porovnání jsou přesně tato část.

Jediné, co je potřeba dotáhnout pro velmi silnou obhajobu, je konzistentní finální tabulka + kvalitativní analýza chyb.

---

## 10) Co se hodí doplnit do BP (není nutně v TODO, ale zlepší obhajobu)

### 10.1 Doporučené doplnění do textu práce

- Přidat sekci „Proč speaker-level nehodnotím u MIX bez diarizace“ (metodická korektnost).
- Přidat „Threats to validity“: jedna nahrávka, doménová závislost, citlivost na VAD parametry.
- Přidat krátkou ablační část: vliv spectral cleaningu vs bez cleaningu.

### 10.2 Doporučené doplnění do výsledků

- Uvést relativní zlepšení proti MIX baseline (např. procentní pokles WER).
- Přidat alespoň jednoduchý interval spolehlivosti (bootstrap přes segmenty), pokud to stihneš.
- Uvést 2–3 párové ukázky transkriptů (MIX vs INDIVIDUAL) se stejným časovým úsekem.

### 10.3 Praktická šablona tvrzení do závěru

„V experimentu na datech X dosáhla stereo větev (INDIVIDUAL + separace) lepší přesnosti než MIX baseline při zachování stejného evaluatoru, stejného eval rozsahu a stejných normalizačních pravidel; tím je doložen praktický přínos využití stereo informace pro ASR v dialogových nahrávkách.“
