# Denoiser (`audio_cleaner_gate.py`)

## Účel
Skript `scripts/audio_cleaner_gate.py` slouží k potlačení přeslechu mezi stereo kanály u nahrávek typu rozhovor.
Cíl je jednoduchý: v každém kanálu ponechat hlavně dominantního mluvčího a slabšího řečníka utlumit.

Důležité vlastnosti:
- pracuje přímo se stereo vstupem (L/R),
- zachovává délku nahrávky (časová osa zůstává stejná),
- neprovádí diarizaci,
- neřeší přepis řeči, pouze úpravu audia.

## Jak skript funguje
Skript zpracovává audio po krátkých rámcích (frame):
1. pro každý rámec porovná energii cílového kanálu vůči druhému kanálu,
2. spočte dominanci v dB,
3. pokud dominance nedosáhne prahu, rámec se výrazně utlumí,
4. výsledek skládá zpět overlap-add metodou (bez zkrácení časové osy).

Poznámka: gate verze nepoužívá explicitní odečet druhého kanálu ani spektrální separaci,
jen pravidla zeslabení podle dominance.

## Vstup a výstup
### Vstup
- 1 argument: cesta ke stereo WAV souboru

Příklad:
`python scripts/audio_cleaner_gate.py data/12008_001.wav`

### Výstupy
- vyčištěný levý kanál: `data/<nazev>_L_gate.wav`
- vyčištěný pravý kanál: `data/<nazev>_R_gate.wav`
- report s metrikami: `results/<nazev>_gate_report.json`

## Hlavní nastavitelné proměnné (v kódu)
Konfigurace je na začátku souboru `scripts/audio_cleaner_gate.py`:

- `OUTPUT_DIR`  
  cílová složka pro vyčištěné WAV soubory.

- `FRAME_MS`  
  délka rámce v milisekundách pro vyhodnocení dominance.

- `HOP_MS`  
  krok mezi rámci v milisekundách.

- `DOMINANCE_DB_CUT`  
  prah dominance v dB; vyšší hodnota = agresivnější potlačení přeslechu.

- `AUTO_DOMINANCE_DB_CUT`  
  zapíná automatický odhad `DOMINANCE_DB_CUT` podle konkrétní nahrávky.

- `AUTO_DOMINANCE_PERCENTILE`, `AUTO_DOMINANCE_MIN_DB`, `AUTO_DOMINANCE_MAX_DB`  
  řídí, jak se automatický práh vypočítá a do jakého rozsahu se omezí.

- `NUM_PASSES`  
  počet průchodů filtrem; více průchodů většinou více utlumí přeslech.

- `MIN_KEEP_GAIN`  
  minimální podíl signálu v nedominantním rámci (0.0 = úplné ticho, vyšší = méně záseků).

## Interpretace reportu
Soubor `results/*_gate_report.json` obsahuje:
- použité parametry (`params`),
- statistiky před/po (`stats`), zejména:
  - `left_rms_before_db` / `left_rms_after_db`,
  - `right_rms_before_db` / `right_rms_after_db`,
  - `corr_before` / `corr_after`.

Prakticky:
- větší pokles RMS po čištění obvykle znamená silnější útlum přeslechu,
- příliš velký útlum může být slyšet jako nepřirozenost nebo přerušování.

## Doporučení pro ladění
- pokud je přeslech stále slyšet: zvyš `DOMINANCE_DB_CUT` nebo `NUM_PASSES`,
- pokud používáš auto-práh a chceš agresivnější výsledek: zvyš `AUTO_DOMINANCE_PERCENTILE`,
- pokud jsou slyšet záseky: zvyš `MIN_KEEP_GAIN`,
- ladit vždy po malých krocích a po každé změně poslechnout několik různých úseků.
