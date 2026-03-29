# Hlavní dokumentace ASR projektu (zkrácená verze)

## 1. O co v projektu jde

Projekt řeší automatický přepis českých interview nahrávek.
Aktuální cíl je mít obhajitelnou baseline, porovnat varianty ASR a připravit separační větev.

Prakticky řešíme:

- MONO baseline (Whisper, Faster-Whisper),
- STEREO baseline bez separace,
- jednotné vyhodnocení přes WER,
- přípravu na separaci a případnou diarizaci po mluvčích.

---

## 2. Co už máme hotové

### 2.1 Skripty

- scripts/asr_mono_whisper.py
- scripts/asr_mono_fastwhisper.py
- scripts/asr_stereo.py
- scripts/evaluate_wer.py

### 2.2 Evaluace

Vyhodnocení běží přes jednotný evaluator.
Používá se strict i robust WER, plus diagnostické reporty.

### 2.3 Důležité pozorování

- Faster-Whisper varianta s chunkingem běží rychleji než klasický Whisper.
- Po sjednocení chunkingu a postprocessingu vychází WER mezi Whisper a Faster-Whisper podobně.
- Chunking pomáhá stabilitě (hlavně náběh s tichem/šumem) a při overlapu drží kvalitu na podobné úrovni.
- Stereo bez separace má výrazný přeslech, což je správná baseline před separací.

---

## 3. Aktuální technický směr

### 3.1 Baseline neměnit zbytečně

Baseline má být stabilní a reprodukovatelná.
Není cílem ji neustále ladit na minimum WER.
Je cílem mít referenční bod pro další kroky.

### 3.2 Výstupní schema

Všechny hlavní výstupy držíme ve stejném JSON stylu:

- metadata
- segments
- full_transcription

Segmenty drží minimálně:

- id
- speaker
- speakers
- start
- end
- text
- is_overlap

To je klíčové pro férové porovnání.

---

## 4. ZIPFORMER – na rozpoznání češtiny: ANO nebo NE?

### Krátká odpověď

ANO, ale jako samostatná experimentální větev.
NE jako náhrada aktuální baseline v této fázi.

### Důvod

- Baseline je potřeba udržet stabilní (Whisper/Faster-Whisper) kvůli porovnatelnosti.
- Zipformer může být velmi dobré rozšíření pro češtinu.
- Pokud ho nasadíme teď místo baseline, ztratíme čisté srovnání.

### Doporučení do práce

Zipformer uvést jako „pokročilou větev“ nebo „future/extended experiment“.
Výsledky porovnat proti stávající baseline stejným evaluátorem.

---

## 5. DIARIZACE – MONO vs STEREO

### Zadání a realita

Aktuálně není cíl dělat plnohodnotnou mono diarizaci jako hlavní větev.
Hlavní praktický směr je STEREO + separace.

### Co chceme metodicky

Vyhodnocovat nejen „celkový text“, ale i po mluvčích.
To znamená, že evaluator by měl umět speaker-level režim.

### Doporučený postup bez zásahu do baseline

Nesahat do baseline skriptů.
Místo toho přidat samostatný skript, například:

- scripts/evaluate_speaker_wer.py

Ten bude:

1. číst standardní JSON výsledky,
2. číst GT s mluvčími,
3. párovat segmenty po speaker label,
4. počítat WER per speaker + celkově.

Tím oddělíme experimentální logiku od baseline pipeline.

---

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

## 7. Co přesně chceme dokázat v BP

1. Baseline bez separace má jasné limity (zejména ve stereo).
2. Změna backendu/modelu (Whisper vs Faster-Whisper) v této fázi přináší hlavně runtime zrychlení, zatímco WER zůstává podobné při srovnatelném chunkingu.
3. Volba strategie zpracování (chunk + overlap) je trade-off stabilita vs rychlost, zatímco WER zůstává podobné.
4. Separace ve stereo větvi zlepší přeslech a následně i kvalitu přepisu.
5. Vyhodnocení po mluvčích dá přesnější obraz, kde pipeline funguje a kde ne.

---

## 8. Praktické rozhodnutí pro nejbližší období

### Nechat stabilní

- asr_mono_whisper.py
- asr_mono_fastwhisper.py
- asr_stereo.py
- evaluate_wer.py

### Přidat navíc

- evaluate_speaker_wer.py (nový skript mimo baseline)
- první separační skript pro stereo větev

### Nespouštět teď

- velký refaktor baseline,
- míchání více experimentálních zásahů najednou,
- změny schema, které by rozbily staré výsledky.

---

## 9. Jak to formulovat do textu práce

V metodice jasně oddělit tři vrstvy:

1. Baseline (mono/stereo bez separace)
2. Model/backend experimenty (Whisper vs Faster-Whisper, případně Zipformer)
3. Pokročilá stereo větev (separace + speaker-level eval)

Takto bude práce přehledná a obhajitelná.

---

## 10. Shrnutí jednou větou

Aktuální stav je připravený na další krok: baseline je funkční a měřitelná, teď má smysl přidat separaci a speaker-level evaluaci v samostatných skriptech bez rozbití toho, co už funguje.


„Mono větev je kontrolní baseline, hlavní hypotéza a hlavní přínos práce jsou ověřovány ve stereo/multikanálové větvi.“