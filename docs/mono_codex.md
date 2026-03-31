# MONO větev (srovnávací): co dál bez rozbití baseline

## 1) Krátká odpověď na otázku

Ano, diarizace na mono se udělat dá.
Není ale tak spolehlivá jako diarizace ve stereo po separaci.

Takže:

- **ANO**: mono diarizace je validní experimentální větev.
- **NE**: není to nejlepší hlavní směr, pokud je primární cíl stereo separace.

Prakticky je nejlepší mít mono diarizaci jako „vedlejší analytickou větev“, ne jako náhradu hlavního cíle.

### Ukotvení vůči zadání BP

Hlavní směr BP je využití stereo/multikanálové informace pro zlepšení ASR.
MONO diarizace je zde pouze kontrolní a srovnávací experiment, aby bylo možné férově porovnat přínos stereo větve.

---

## 2) Proč je mono diarizace těžší

U mono signálu máš jen jeden kanál.
Nemáš prostorovou informaci (levý/pravý rozdíl).
Model musí rozhodovat mluvčí jen z hlasových vlastností a časového průběhu.

To vede k typickým problémům:

- horší rozlišení při překryvu řeči,
- častější záměna mluvčích v hraničních úsecích,
- větší citlivost na kvalitu záznamu,
- občasná fragmentace mluvčího do více pseudo-mluvčích.

To není chyba přístupu, je to vlastnost úlohy.

---

## 3) Co jde na mono udělat smysluplně hned

Níže jsou kroky, které dávají metodicky i prakticky smysl, aniž bys musel měnit baseline skripty.

### 3.1 Přidat samostatný mono diarizační skript

Doporučení:

- nový soubor `scripts/mono_diarization.py`
- vstup: mono wav + (volitelně) ASR segmenty
- výstup: segmenty s přiřazeným `speaker` / `speakers`

Důležité: držet stejný JSON styl jako ostatní výstupy.

### 3.2 Přidat samostatný evaluator po mluvčích

Doporučení:

- nový soubor `scripts/evaluate_speaker_wer.py`
- nevrtat do `evaluate_wer.py` baseline režimu

Co má dělat:

1. načíst GT se speakery,
2. načíst hypotézu se speakery,
3. spočítat WER per speaker,
4. spočítat agregovaný speaker-aware WER,
5. vygenerovat diagnostiku po mluvčích.

Tohle je velmi dobré do bakalářky, protože ukážeš nejen „kolik chyb“, ale i „u koho“.

### 3.3 Držet mono diarizaci jako experimentální větev

Ve výsledcích to popsat jako:

- baseline text-level WER,
- mono speaker-aware WER (experiment),
- stereo+separace speaker-aware WER (hlavní směr).

Tím je práce logicky čistá.

Praktické pravidlo rozsahu:

- mono větev držet „minimal viable“ (pilot + vyhodnocení),
- nerozšiřovat ji na úkor stereo separační větve.

---

## 4) Co nedělat teď (aby ses neztratil)

- Neměnit stabilní baseline skripty každou iteraci.
- Nemíchat mono diarizaci přímo do baseline ASR skriptů.
- Nezavádět najednou 5 nových knihoven bez srovnávacího plánu.
- Nehodnotit diarizaci jen subjektivně bez metrik.

---

## 5) Doporučené mono experimenty (praktické)

## Experiment A – „ASR first, diarization later"

1. Udělat standardní mono ASR (co už máš).
2. Samostatně udělat diarizační timeline.
3. Zarovnat ASR segmenty na diarizační segmenty přes čas.
4. Vyhodnotit text i po mluvčích.

Výhoda:

- minimální zásah do existujícího kódu,
- dobrá kontrola pipeline,
- snadná reprodukovatelnost.

## Experiment B – „Diarization first, ASR per speaker segment"

1. Nejdřív rozdělit mono timeline na speaker segmenty.
2. Každý segment přepsat zvlášť.
3. Složit výstup zpět do jednotného JSON.
4. Vyhodnotit.

Výhoda:

- může snížit míchání textu dvou mluvčích.

Nevýhoda:

- náchylné na diarizační chyby,
- může být pomalejší a fragmentovanější.

## Experiment C – „Hybrid"

1. Běžný ASR běh.
2. Dodat speaker label až v postprocess kroku.
3. Porovnat proti A/B.

Tohle je často nejlepší kompromis do BP.

---

## 6) Jak to napsat do práce (bez chaosu)

Doporučené členění:

### 6.1 Baseline kapitola

- mono whisper/faster-whisper,
- stereo baseline bez separace,
- text-level WER.

### 6.2 Rozšířená mono větev

- mono diarizace jako experiment,
- speaker-aware evaluace,
- limity mono diarizace.

### 6.3 Hlavní pokročilá větev

- stereo separace,
- ASR po separaci,
- speaker-aware evaluace,
- porovnání proti mono i stereo baseline.

Takto zůstane struktura práce jasná.

---

## 7) Je mono větev ještě vůbec užitečná, když je cíl stereo?

Ano, je užitečná ze tří důvodů:

1. Je to referenční spodní hranice (co zvládne systém bez separace).
2. Umožní ukázat, že Faster-Whisper (se stejným chunkingem) typicky zrychlí běh při podobné WER.
3. Umožní ukázat, že speaker-aware evaluace dává smysl už před separací.

Jinými slovy: mono není slepá větev.
Je to důležitý mezikrok a analytický základ.

---

## 8) Konkrétní odpověď „co dál s mono"

Pokud nechceš měnit baseline, pokračuj takto:

1. Nech `asr_mono_whisper.py` a `asr_mono_fastwhisper.py` tak, jak fungují.
2. Přidej nový skript `mono_diarization.py`.
3. Přidej nový skript `evaluate_speaker_wer.py`.
4. Udělej 2–3 kontrolní běhy na stejném časovém rozsahu.
5. Zapiš tabulku: text-WER vs speaker-WER.

Tohle je čistý, obhajitelný a praktický postup.

---

## 9) Návrh minimálního plánu na příští sprint

- [x] vytvořit kostru `scripts/mono_diarization.py`
- [ ] vytvořit kostru `scripts/evaluate_speaker_wer.py`
- [ ] definovat jednotný speaker mapping (`interviewer`, `interviewee`, `unknown`)
- [ ] spustit pilotní mono diarization experiment
- [ ] vyhodnotit per-speaker WER
- [x] sepsat krátké shrnutí do dokumentace
- [ ] po pilotu MONO větev uzavřít jako srovnávací baseline a vrátit fokus na STEREO

---

## 10) Shrnutí jednou větou

Mono diarizace smysl má, ale jen jako doplňková srovnávací větev; hlavní cíl práce zůstává stereo separace, kde má být hlavní implementační i evaluační důraz.

---

## 11) TODO list (MONO + srovnání)

### 11.1 Hlavní cíl práce (ukotvení)

Cíl práce: ukázat, že stereo informace přináší lepší rozlišení jednotlivých mluvčích než mono.
Mono větev je referenční základ a kontrolní experiment.

### 11.2 TODO – MONO větev

- [x] Vygenerovat finální MONO výstup na stejném časovém rozsahu jako STEREO (pro férovost).
- [x] Uložit WER strict/robust pro referenční MONO variantu do tabulky.
- [x] Uložit i runtime pro referenční MONO variantu.
- [x] Přidat krátké kvalitativní ukázky typických MONO chyb (2–3 příklady).

### 11.3 TODO – MONO speaker-level (experiment)

- [ ] Připravit `scripts/mono_diarization.py` jako samostatný experimentální skript.
- [ ] Připravit `scripts/evaluate_speaker_wer.py` bez zásahu do baseline evaluatoru.
- [ ] Definovat mapování labelů (`interviewer`, `interviewee`, `unknown`) a držet ho konzistentně.
- [ ] Spočítat pilotní speaker-level WER pro MONO (aspoň 1 rozsah).
- [ ] Zapsat limity MONO diarizace (překryv, záměna mluvčího, fragmentace).
- [ ] Uložit per-speaker metriky minimálně: WER, počet slov, počet segmentů, hlavní chyby (sub/ins/del).

### 11.4 TODO – Připrava pro finální srovnání

- [ ] Připravit jednotný CSV/Markdown souhrn výsledků pro 3 hlavní varianty: mono reference, stereo baseline (bez separace), stereo + separace.
- [ ] U každé varianty evidovat: model, backend, rozsah, runtime, WER strict, WER robust.
- [ ] U speaker-aware variant doplnit: WER per speaker + agregovaný speaker-aware WER.
- [ ] U stereo + separace evidovat navíc metadata separace (metoda/model, klíčové parametry, verze skriptu).
- [ ] Přidat error breakdown (top sub/ins/del) pro stereo baseline vs stereo + separace.
- [ ] Připravit krátkou textovou interpretaci „co zlepšil model“ vs „co zlepšila prostorová informace“.

### 11.5 TODO – Eval protokol a kritérium úspěchu (speaker-aware)

- [ ] Zamknout eval protokol pro všechny varianty: stejné GT, stejné časové okno, stejné normalizace textu, stejný evaluator.
- [ ] Definovat hlavní cíl vyhodnocení: speaker-aware zlepšení stereo + separace proti stereo baseline.
- [ ] Definovat sekundární cíl: stereo + separace má být minimálně srovnatelné nebo lepší než MONO reference ve speaker-aware metrice.
- [ ] Definovat „done“ kritérium do BP: kompletní tabulka + interpretace + 2–3 ukázky typických chyb před/po separaci.
