# Obnova prostředí (Conda)

Tento projekt má 3 snapshot soubory prostředí v kořeni workspace:

- `environment.yml` – přenositelné (Windows/Linux) minimální prostředí.
- `requirements-conda-explicit.txt` – přesný lock export (`conda list --explicit`) z konkrétní platformy.
- `requirements-pip-freeze.txt` – pip balíčky (`pip freeze`) pro kontrolu/porovnání.


## Přesná obnova (co nejvíc 1:1)

Poznámka: `requirements-conda-explicit.txt` je platform-specific (aktuálně `win-64`) a typicky nepůjde použít 1:1 na Linuxu.

```powershell
conda create -n bakalarka_exact --file requirements-conda-explicit.txt
conda activate bakalarka_exact
```

## Linux doporučení

Na Linuxu používej primárně `environment.yml`:

```bash
conda env create -f environment.yml
conda activate bakalarka_restore
```

## Pravidelná aktualizace snapshotu

Z kořene projektu spusť:

```powershell
conda env export -n bakalarka --no-builds | Out-File -Encoding utf8 environment.yml
conda list -n bakalarka --explicit | Out-File -Encoding utf8 requirements-conda-explicit.txt
conda run -n bakalarka pip freeze | Out-File -Encoding utf8 requirements-pip-freeze.txt
```
