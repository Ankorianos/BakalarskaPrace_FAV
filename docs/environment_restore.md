# Obnova prostředí (Conda)

Tento projekt má 3 snapshot soubory prostředí v kořeni workspace:

- `environment.yml` – přenositelný export Conda prostředí bez build stringů.
- `requirements-conda-explicit.txt` – přesný lock export (`conda list --explicit`).
- `requirements-pip-freeze.txt` – pip balíčky (`pip freeze`) pro kontrolu/porovnání.

## Doporučená obnova (portable)

```powershell
conda env create -n bakalarka_restore -f environment.yml
conda activate bakalarka_restore
```

## Přesná obnova (co nejvíc 1:1)

```powershell
conda create -n bakalarka_exact --file requirements-conda-explicit.txt
conda activate bakalarka_exact
```

## Pravidelná aktualizace snapshotu

Z kořene projektu spusť:

```powershell
conda env export -n bakalarka --no-builds | Out-File -Encoding utf8 environment.yml
conda list -n bakalarka --explicit | Out-File -Encoding utf8 requirements-conda-explicit.txt
conda run -n bakalarka pip freeze | Out-File -Encoding utf8 requirements-pip-freeze.txt
```
