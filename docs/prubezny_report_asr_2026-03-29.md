# Průběžný report – ASR pipeline (stav k 2026-03-29)

## 1) Cíl práce v aktuální fázi

Cílem této fáze bylo stabilizovat a porovnat více ASR variant nad stejnými daty a stejným eval rozsahem.
Hlavní důraz byl na:

- baseline bez separace,
- reprodukovatelnost výstupu,
- férové srovnání variant (stejný časový rozsah, stejný evaluator),
- praktickou robustnost (zejména problém ticha/šumu na začátku úseku).

V této fázi ještě není cílem perfektní diarizace ani finální separační pipeline.
Smyslem je mít pevný referenční bod před separací.

---

## 2) Aktuální struktura skriptů

V adresáři scripts jsou aktuálně aktivní tyto ASR skripty:

- asr_mono_whisper.py
- asr_mono_fastwhisper.py
- asr_stereo.py
- evaluate_wer.py

### 2.1 asr_mono_whisper.py

Whisper (openai/whisper) varianta s chunkingem a postprocessingem.
Obsahuje:

- chunk decode,
- overlap,
- dedup/merge logiku,
- časové okno start/end,
- výstup do JSON struktury kompatibilní s evaluátorem.

Tato varianta je robustní, ale má známý boundary efekt na hranách chunků.

### 2.2 asr_mono_fastwhisper.py

Faster-Whisper varianta s chunkingem a postprocessingem, metodicky sladěná s whisper větví.
Cíl této větve je porovnat hlavně backend/model při podobné segmentační strategii.

V aktuálním stavu:

- model: podle nastavení v souboru,
- transcribe nad celým vybraným úsekem,
- kompatibilní JSON výstup,
- metadata obsahují backend/model/device/compute_type.

Poznámka: při běhu na Windows byl řešen OpenMP konflikt (libomp vs libiomp).

### 2.3 asr_stereo.py

Stereo split baseline (L/R), bez separace zdrojů.
Byl upraven tak, aby měl lepší logiku než původní velmi jednoduchá verze a aby byl formát kompatibilní s eval pipeline.

Důležitá vlastnost baseline:

- v obou kanálech je stále slyšet přeslech druhého mluvčího,
- to je očekávané a metodicky správné jako baseline před separací,
- díky tomu bude pozdější separační pipeline lépe obhajitelná (jasně ukáže přínos).

---

## 3) Evaluace (WER) – aktuálně dostupná čísla

Níže jsou hodnoty, které jsou aktuálně vygenerované v diagnostics souborech.

### 3.1 MONO Whisper baseline

Zdroj:
- results/mono_results_range.json

Diagnostics:
- results/wer_diagnostics_mono_results_range.txt

Hodnota:
- ROBUST WER: 19.03 %

### 3.2 MONO Faster-Whisper

Zdroj:
- results/mono_results_range_fastwhisper.json

Diagnostics:
- results/wer_diagnostics_mono_results_range_fastwhisper.txt

Hodnota:
- ROBUST WER: viz aktuální diagnostics soubor (pro finální porovnání používat běhy se stejným chunkingem jako u whisper větve)

### 3.3 MONO interpretace

Po sjednocení chunkingu a postprocessingu mezi větvemi vychází Faster-Whisper primárně jako runtime zrychlení.

Interpretace:

- rozdíl ve WER mezi whisper a faster-whisper je v této fázi podobný,
- hlavní praktický přínos Faster-Whisper je rychlost běhu.

Pro čistou metodiku je stále vhodné oddělovat:

1) vliv modelu/backendu,
2) vliv segmentační strategie.

### 3.4 STEREO baseline (bez separace)

Zdroj:
- results/stereo_results_range.json

Diagnostics:
- results/wer_diagnostics_stereo_results_range.txt

Hodnota:
- ROBUST WER: 94.03 %

Interpretace:

Tato hodnota je očekávaně velmi špatná kvůli překryvu obsahu mezi kanály a vysokému počtu insercí.
Jako baseline před separací je to v pořádku.

---

## 4) Ticho/šum na začátku – poznatky

V průběhu experimentů se ukázalo:

- začátek úseku s tichem nebo šumem může spouštět halucinační text,
- chunking někdy pomáhá stabilitě startu,
- zároveň chunking může zvyšovat boundary chyby na hranách segmentů.

Praktický kompromis:

- zachovat stabilní konfiguraci, která nepadá a drží rozumný WER,
- nespouštět velké redesigny před separační fází,
- změny dělat jen pokud mají jasný měřitelný přínos.

---

## 5) Proč mít dvě MONO varianty

Dvě větve (Whisper vs Faster-Whisper) jsou metodicky užitečné:

- lze doložit, že zlepšení nemusí být jen postprocessing trik,
- lze ukázat vliv backendu/modelu,
- lze transparentně popsat, že Faster-Whisper zrychluje běh při podobném WER.

To je vhodné i do textu práce jako mini-ablation.

---

## 6) Doporučené prezentování v BP

Do metodiky/výsledků se hodí explicitně uvést:

- baseline A: mono whisper chunked,
- baseline B: mono faster-whisper chunked,
- baseline C: stereo split bez separace,
- separační větev: bude následovat jako rozšíření nad C.

Tím je jasně odděleno:

- co je základ,
- co je zlepšení modelem,
- co je zlepšení separací.

---

## 7) Praktický plán dalších kroků

1. Zamknout aktuální parametry MONO variant (neměnit je každou iteraci).
2. Vygenerovat finální sadu WER čísel pro stejný rozsah (180–360) pro všechny baseline větve.
3. Začít separační větev (L/R separace → ASR → merge) se stejným output schema.
4. Vyhodnotit separační větev stejným evaluate_wer.py.
5. Doplnit tabulku: varianta, WER strict, WER robust, runtime, poznámka o stabilitě.

6. Diarizace a tak - KUBA

---

## 8) Co je teď hotové

- sjednocené výstupy JSON pro evaluaci,
- robustní evaluator,
- funkční MONO whisper větev,
- funkční MONO faster-whisper větev,
- funkční stereo baseline větev,
- dostupné diagnostické reporty pro všechny hlavní větve.

To znamená, že projekt má už teď solidní experimentální základ pro separační fázi.

---

## 9) Krátké shrnutí jednou větou

Aktuální stav je dostatečně stabilní a metodicky obhajitelný: MONO baseline funguje, Faster-Whisper při srovnatelném chunkingu hlavně zrychluje runtime při podobném WER, a stereo baseline bez separace jasně ukazuje problém přeslechu, který má další fáze cíleně řešit.
