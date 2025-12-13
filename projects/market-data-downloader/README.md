# market-data-downloader

Projet (mode script) pour télécharger des données de marché.

Objectifs v1:

- Télécharger des **OHLCV** depuis plusieurs sources (ex: Binance Futures, Hyperliquid Perps)
- Stockage standardisé en **Parquet** à la racine du monorepo
- Éviter le re-téléchargement (cache par fichiers mensuels)
- UX par projet (CLI interactive + mode non-interactif)

## Stockage des données

Les données sont stockées (hors Git) dans:

- `data/market_data/<dataset>/<source>/<symbol>/<timeframe>/YYYY/MM.parquet`

Datasets v1:

- `ohlcv` (seul implémenté pour le moment)

Schéma v1 (Parquet):

- `timestamp_ms` (int64, UTC, epoch ms)
- `open`, `high`, `low`, `close`, `volume` (float64)

## Démarrage rapide (Windows)

Depuis `projects/market-data-downloader/`:

```bash
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Interface web (Streamlit) (UX principale)

Depuis `projects/market-data-downloader/`:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## CLI (fallback / mode avancé)

Lister les sources:

```bash
python src\main.py sources
```

Télécharger des données:

```bash
python src\main.py download --source hyperliquid_perps --symbol BTC --tf 1h --start 2025-07-01 --end 2025-07-07
```

## Test rapide (E2E)

### Hyperliquid (Perps)

Télécharger un petit range puis relancer la même commande pour vérifier le cache (pas de re-téléchargement):

```bash
python src\main.py download --source hyperliquid_perps --symbol BTC --tf 1h --start 2025-07-01 --end 2025-07-07
python src\main.py download --source hyperliquid_perps --symbol BTC --tf 1h --start 2025-07-01 --end 2025-07-07
```

### Binance Futures

Le plus simple est de lancer le mode interactif (choix source/symbol/timeframe dans des menus):

```bash
python src\main.py
```

## Utilisation

### Mode interactif

```bash
python src\main.py
```

### Mode non-interactif

```bash
python src\main.py download --dataset ohlcv --source binance_futures --symbol BTC/USDT --timeframe 1h --start 2025-01-01 --end 2025-12-31
```

## Secrets

- Aucune clé n’est requise pour OHLCV public.
- Ne jamais commiter de `.env`.
- Utiliser `.env.example` comme modèle si besoin.
