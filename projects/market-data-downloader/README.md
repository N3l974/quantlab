# market-data-downloader

Projet (mode script) pour télécharger des données de marché.

Objectifs v1:

- Télécharger des **OHLCV** depuis plusieurs sources (ex: Binance Futures, Hyperliquid Perps, Kraken Spot, Kraken Futures)
- Stockage standardisé en **Parquet** à la racine du monorepo
- Éviter le re-téléchargement (cache par fichiers mensuels)
- UX par projet (CLI interactive + mode non-interactif)

## Sources supportées (v1)

Noms des sources (CLI/Streamlit):

- `binance_futures`
- `hyperliquid_perps`
- `kraken_spot`
- `kraken_futures`
- `mt5_icmarkets`

## cTrader Open API (en pause / reprise)

Objectif: ajouter une source `ctrader_icmarkets` (demo/live) basée sur cTrader Open API pour télécharger des OHLCV (trendbars).

### Pré-requis

- Avoir un compte cTrader (cTID) + un compte broker (ex: ICMarkets demo/live).
- Créer une application sur le portail Open API: https://openapi.ctrader.com/
- Attendre la validation/activation de l'app si requis (certaines apps restent en statut pending un moment).

### Checklist OAuth (obtenir un access token)

1) Dans l'app Open API:

- Définir un `redirect_uri` (ex: `http://localhost:8787/callback`).
- Récupérer `client_id` et `client_secret`.

2) Ouvrir l'URL d'autorisation (dans un navigateur) puis cliquer "Allow access".

- Scope recommandé pour downloader OHLCV: `accounts` (pas besoin de trading).

```text
https://id.ctrader.com/my/settings/openapi/grantingaccess/?client_id={client_id}&redirect_uri={redirect_uri}&scope=accounts&product=web
```

3) Récupérer le paramètre `code` dans l'URL de redirection:

```text
{redirect_uri}?code=XXXX
```

4) Échanger `code` contre `accessToken` et `refreshToken`:

```text
https://openapi.ctrader.com/apps/token?grant_type=authorization_code&code={code}&redirect_uri={redirect_uri}&client_id={client_id}&client_secret={client_secret}
```

5) Identifier le `ctidTraderAccountId` (compte demo/live) via la requête Open API "GetAccountListByAccessToken" (nécessite une connexion au proxy Open API), puis l'utiliser pour authentifier la session.

### Endpoints (proxy Open API)

- Protobuf:
  - Demo: `demo.ctraderapi.com:5035`
  - Live: `live.ctraderapi.com:5035`
- JSON:
  - Demo: `demo.ctraderapi.com:5036`
  - Live: `live.ctraderapi.com:5036`

### Variables d'environnement (prévu pour l'intégration)

La source `ctrader_icmarkets` n'est pas encore implémentée, mais l'intégration utilisera des variables d'environnement (pas de secrets en dur):

- `CTRADER_CLIENT_ID`
- `CTRADER_CLIENT_SECRET`
- `CTRADER_REDIRECT_URI`
- `CTRADER_ACCESS_TOKEN`
- `CTRADER_REFRESH_TOKEN`
- `CTRADER_ACCOUNT_ID` (ctidTraderAccountId)
- `CTRADER_ENV` = `demo` | `live`

## Stockage des données

Les données sont stockées (hors Git) dans:

- `data/market_data/<dataset>/<source>/<symbol>/<timeframe>/YYYY/MM.parquet`

Datasets v1:

- `ohlcv` (seul implémenté pour le moment)

Notes:

- Certaines sources ont des limites d'historique (ex: Hyperliquid ~5000 candles) : l'UI peut afficher une date de départ estimée.
- Un ancien layout sans le dossier `ohlcv/` peut exister; le downloader peut migrer au nouveau layout lors d'un téléchargement.

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
python src\main.py download --dataset ohlcv --source hyperliquid_perps --symbol BTC --tf 1h --start 2025-07-01 --end 2025-07-07
```

## Test rapide (E2E)

### Hyperliquid (Perps)

Télécharger un petit range puis relancer la même commande pour vérifier le cache (pas de re-téléchargement):

```bash
python src\main.py download --dataset ohlcv --source hyperliquid_perps --symbol BTC --tf 1h --start 2025-07-01 --end 2025-07-07
python src\main.py download --dataset ohlcv --source hyperliquid_perps --symbol BTC --tf 1h --start 2025-07-01 --end 2025-07-07
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
- `mt5_icmarkets` utilise le terminal MT5 local (login effectué dans le terminal) et ne nécessite pas de clés API.
- cTrader Open API nécessite des tokens OAuth (à fournir via variables d'environnement).
- Ne jamais commiter de `.env`.
- Utiliser `.env.example` comme modèle si besoin.
