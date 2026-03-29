# STEREO codex: co dělat dál (separace, ASR, speaker-level evaluace)

## 1) Smysl stereo větve

Stereo větev má v projektu zásadní roli.
V baseline bez separace je v kanálech přeslech, což zhoršuje přepis i WER.
Právě proto dává stereo separace metodicky i prakticky smysl.

Krátká odpověď:

- Ano, stereo větev má smysl.
- Ano, speaker-level evaluace má smysl.
- Ano, je vhodné ji dělat jako samostatný krok mimo baseline.

Poznámka k ASR backendu:

- při stejném chunkingu vychází Faster-Whisper obvykle rychlejší,
- WER zůstává v této fázi podobné jako u klasického Whisper.

---

## 2) Hlavní plán další fáze

## 6. STEREO plán: separace → ASR → evaluate → diarizace

Toto je hlavní plán další fáze:

1. Vstup: stereo (L/R).
2. Separace: redukovat přeslech mezi mluvčími.
3. ASR: přepis každé separované stopy.
4. Merge: sjednocení do stejného JSON schema.
5. Evaluate WER: stejný evaluator jako dosud.
6. Speaker-level evaluate: samostatný skript po mluvčích.

### Proč takto

- zachováme čistou návaznost na baseline,
- jasně ukážeme přínos separace,
- budeme mít měření jak celkově, tak po mluvčích.

---

## 3) Co přesně znamená speaker-level evaluate

Speaker-level evaluace znamená, že WER nepočítáš jen nad celým textem, ale i po jednotlivých mluvčích.
To je důležité, protože agregované WER může skrýt, že jeden mluvčí je přepisovaný výrazně hůř.

Doporučený výstup:

- celkový WER,
- WER pro speaker A,
- WER pro speaker B,
- počet slov pro každého mluvčího,
- základní diagnostika substitucí/insercí/delecí po speakerovi.

---

## 4) Jak to implementovat bez zásahu do baseline

Aby se neporušila stabilní baseline, doporučený postup je přidat nové skripty.

Doporučené nové soubory:

- scripts/asr_stereo_separation.py
- scripts/evaluate_speaker_wer.py

### 4.1 asr_stereo_separation.py

Úkol skriptu:

1. Načíst stereo vstup.
2. Udělat separaci (v první verzi klidně jednoduchou, ale reprodukovatelnou).
3. Přepsat separované stopy stejným ASR backendem jako baseline.
4. Sloučit do stejného JSON schema.
5. Uložit metadata separace (model, parametry, rozsah, verze).

### 4.2 evaluate_speaker_wer.py

Úkol skriptu:

1. Načíst GT s mluvčími.
2. Načíst hypotézu se speakery.
3. Udělat mapování speaker labelů.
4. Spočítat WER celkově i po mluvčích.
5. Uložit diagnostiku do results/.

---

## 5) Doporučený postup po krocích

### Krok 1: Zamknout baseline

Nesahat do:

- scripts/asr_stereo.py
- scripts/evaluate_wer.py

Baseline musí zůstat stabilní pro porovnání.

### Krok 2: Udělat první separační běh

Nejdřív stačí jedna pracovní verze.
Cíl není perfektní kvalita, cíl je funkční pipeline se stejným schema.

### Krok 3: Přidat speaker-level evaluaci

Po prvním funkčním separačním výstupu okamžitě doplnit speaker-level metriky.
Bez toho nebude jasné, kdo se zlepšil a kde zůstaly chyby.

### Krok 4: Udělat srovnávací tabulku

Minimální tabulka:

- stereo baseline bez separace,
- stereo + separace,
- stereo + separace + speaker-level report.

Pole v tabulce:

- WER strict,
- WER robust,
- WER per speaker,
- runtime,
- poznámka o kvalitě výstupu.

---

## 6) Co čekat realisticky

Po separaci obvykle:

- klesnou inserce způsobené přeslechem,
- zlepší se čitelnost segmentů,
- agregovaný WER by měl jít dolů,
- speaker-level WER ukáže, jestli se zlepšení týká obou mluvčích.

Nemusí se zlepšit všechno.
Někdy separace pomůže jednomu speakerovi a druhému méně.
Proto je speaker-level evaluace důležitá.

---

## 7) Rizika a jak je řídit

### Riziko 1: Přehnaný zásah do baseline

Mitigace:

- držet nové věci v nových skriptech,
- baseline jen číst, neměnit.

### Riziko 2: Nejasné mapování speakerů

Mitigace:

- explicitní mapovací pravidla,
- uložení mappingu do metadata/logu.

### Riziko 3: Příliš variant najednou

Mitigace:

- nejdřív jedna separační konfigurace,
- potom až další porovnání.

---

## 8) Co psát do BP k této fázi

Do metodiky:

- proč je stereo baseline bez separace důležitá,
- proč má separace očekávaný přínos,
- jak je zavedená speaker-level evaluace.

Do výsledků:

- tabulka před/po separaci,
- krátká interpretace po mluvčích,
- 2–3 konkrétní příklady typických chyb.

Do diskuse:

- kde separace pomohla,
- kde nepomohla,
- jaký je další krok (lepší separační model, robustnější mapping speakerů).

---

## 9) Krátké rozhodnutí pro projekt

Doporučení pro pokračování:

1. Pokračovat stereo směrem.
2. Přidat separační větev mimo baseline.
3. Přidat speaker-level evaluator.
4. Udržet jednotné JSON schema a jednotné vyhodnocení.

To je nejčistší cesta, jak získat obhajitelný výsledek do bakalářky.

---

## 10) Shrnutí jednou větou

Stereo separace + speaker-level evaluace je logický a metodicky správný další krok, který navazuje na současnou baseline a má nejvyšší šanci ukázat měřitelné zlepšení tam, kde baseline selhává.

---

## 11) TODO list (STEREO + finální srovnání)

### 11.1 Hlavní cíl práce (ukotvení)

Cíl práce: prokázat, že stereo informace (a následná separace) přináší lepší rozpoznání jednotlivých mluvčích než mono přístup.

### 11.2 TODO – STEREO baseline

- [ ] Zamknout baseline `scripts/asr_stereo.py` bez dalších zásahů.
- [ ] Vygenerovat referenční `stereo_results_range.json` na stejném rozsahu jako MONO.
- [ ] Uložit baseline WER strict/robust + diagnostiku.
- [ ] Zapsat typické chyby baseline (přeslech, inserce, duplicitní obsah mezi kanály).

### 11.3 TODO – STEREO separační větev

- [ ] Vytvořit `scripts/asr_stereo_separation.py` (samostatně, bez změn baseline skriptu).
- [ ] Zvolit první separační konfiguraci (jedna reprodukovatelná varianta jako start).
- [ ] Přepsat separované stopy stejným ASR backendem jako baseline (pro férovost).
- [ ] Sloučit výstup do stejného JSON schema (`metadata`, `segments`, `full_transcription`).
- [ ] Uložit metadata separace (model, parametry, verze, rozsah).

### 11.4 TODO – Speaker-level evaluace

- [ ] Dokončit `scripts/evaluate_speaker_wer.py` jako samostatný evaluator.
- [ ] Počítat WER po mluvčích i celkově.
- [ ] Přidat diagnostiku po mluvčích (sub/ins/del).
- [ ] Ověřit konzistentní mapování speaker labelů napříč GT a hypotézou.

### 11.5 TODO – Finální porovnání MONO vs STEREO

- [ ] Připravit finální srovnávací tabulku: MONO baseline, MONO fast, STEREO baseline, STEREO separace.
- [ ] Uvést u všech variant stejný eval rozsah a stejné metriky.
- [ ] Vyhodnotit, zda stereo separace snížila inserce způsobené přeslechem.
- [ ] Vyhodnotit, zda stereo zlepšilo speaker-level WER proti mono.
- [ ] Sepsat závěr v jedné větě: „co zlepšil model“ vs „co zlepšila stereo informace“.

### 11.6 TODO – Text do BP

- [ ] Doplnit metodickou část (proč baseline, proč separace, proč speaker-level).
- [ ] Doplnit výsledkovou část (tabulka + krátká interpretace).
- [ ] Doplnit diskusi limitů (kde separace nepomohla nebo pomohla jen částečně).
