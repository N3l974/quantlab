# ohlcv-optimizer-v1

Optimiseur (v1) de stratégies sur données **OHLCV Hyperliquid Perps**, avec interface web (Streamlit).

Objectif:
- sélectionner un instrument / timeframe / période
- lancer une optimisation multi-stratégies
- optimiser les paramètres de stratégie + paramètres communs (TP/SL + mode `none|grid|martingale`)
- scoring **Pareto** (return vs drawdown)
- sélectionner un "champion" sur le **test**: *max return sous `DD <= X`*
- exporter un manifest JSON pour paper trading (hors infra)

## Prérequis

- Windows
- Python 3.11+ recommandé
- Données OHLCV déjà téléchargées via `projects/market-data-downloader/`

## Installation (Windows)

Depuis `projects/ohlcv-optimizer-v1/`:

### 1) Créer un venv

```bash
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

Si PowerShell bloque l'activation:

```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### 2) Installer les dépendances

```bash
pip install -r requirements.txt
```

## Lancer l'interface web (launcher / visualizer)

Depuis `projects/ohlcv-optimizer-v1/` (dans le venv activé):

```bash
python -m streamlit run streamlit_app.py
```

L'optimisation est exécutée dans un **process séparé (CLI headless)** et Streamlit sert à:
- configurer et lancer un run
- visualiser l'état d'un run en cours (même après refresh)
- analyser les runs terminés (Analyze)

## Données attendues

Le projet lit les fichiers Parquet créés par `market-data-downloader`.

Layout attendu:

```text
data/market_data/
  ohlcv/
    hyperliquid_perps/
      <symbol>/
        <timeframe>/
          YYYY/
            MM.parquet
```

Exemple:

```text
data/market_data/ohlcv/hyperliquid_perps/BTC/5m/2025/07.parquet
```

Colonnes Parquet attendues:
- `timestamp_ms` (int64 UTC epoch ms)
- `open`, `high`, `low`, `close`, `volume` (float64)

Sources typiques (selon les données téléchargées):
- `hyperliquid_perps` (crypto perps)
- `mt5_icmarkets` (MT5 local, ex: ICMarkets Raw Spread)

## Utilisation (dans l'UI)

### Dataset
- **Source**: `hyperliquid_perps`
- **Timeframe**: `5m` par défaut (modifiable)
- **Symbol**: liste auto-détectée depuis le dossier data

Notes:
- Le menu **Source** liste les dossiers présents sous `data/market_data/ohlcv/`.
- Pour `mt5_icmarkets`, les symboles/timeframes disponibles dépendent de ce que tu as téléchargé via `projects/market-data-downloader/`.

### Période
- Optionnel: `Start` et `End` (format `YYYY-MM-DD`)
- Si vide: toute la période des Parquets est utilisée

### Stratégies
5 stratégies "signal-only" sont incluses:
- `ma_cross`
- `rsi_reversion`
- `bollinger_breakout`
- `stochastic`
- `range_breakout`

Tu peux activer/désactiver les stratégies dans l'UI.

### Backtest
- **Initial equity**: capital de départ
- **Fee (taker) bps**: par défaut `4.5` bps (Hyperliquid Perps tier 0)
- **Slippage bps**: slippage/spread approximatif (à calibrer)
- **Position management**:
  - `none`: pas d'adds
  - `grid`: adds espacés depuis le dernier fill, TP/SL recalculés sur le prix moyen
  - `martingale`: taille augmente après perte, reset après gain
- **Selection DD threshold (test) %**: `X` pour le filtre de sélection champion

Notes (brokers FX/CFD type MT5):
- `fee_bps` et `slippage_bps` sont une approximation des coûts d'exécution.
- Sur ICMarkets **Raw Spread**, une commission existe (par lot) et peut être approximée en bps.
- Point de départ (ordre de grandeur, à calibrer): `fee_bps ~= 0.35` et `slippage_bps ~= 1.0..3.0`.

## cTrader (en pause / reprise)

L'objectif est de basculer vers une source cTrader (ex: `ctrader_icmarkets`) pour l'exécution live et/ou le téléchargement OHLCV via cTrader Open API.

Statut:
- En attente de validation/activation de l'application sur le portail Open API.
- Les étapes OAuth, endpoints et variables d'environnement sont documentés côté downloader: `projects/market-data-downloader/README.md`.

### Budget d'optimisation
- **Max trials**: nombre max d'essais (par stratégie)
- **Time budget (seconds)**: optionnel (0 = désactivé)
- **Worker processes (SQLite)**: nombre de process OS par stratégie (chaque worker fait des trials avec `n_jobs=1`)

## Exécution headless (CLI)

Chaque run est persisté dans `runs/<run_id>_.../` et contient notamment:
- `context.json`: dataset + config
- `optuna.db`: storage SQLite Optuna
- `status.json`: état courant (pour Streamlit)
- `progress.jsonl`: event log (progress + best-so-far)
- `report.json` + `*.csv`: résultats exportés en fin de run

### Lancer une optimisation depuis le terminal

```bash
python -m hyperliquid_ohlcv_optimizer.optimize.run_optimize --run-dir "runs/<RUN_DIR>" --workers 8
```

### Stopper un run depuis le terminal

```bash
python -m hyperliquid_ohlcv_optimizer.optimize.stop_run --run-dir "runs/<RUN_DIR>" --reason stop
```

Notes:
- `Ctrl+C` dans le terminal du runner écrit un `stop.flag` et arrête proprement.
- Le runner imprime la progression (par stratégie + total) et le **best-so-far** (Pareto train).

## Logique d'optimisation (résumé)

- Split chronologique **train/test = 75% / 25%**
- Optimisation sur **train** en multi-objectif:
  - maximize `return_train_pct`
  - minimize `dd_train_pct`
- On récupère le **front Pareto** et on évalue les candidats sur **test**
- Sélection champion: **max `return_test_pct`** sous `dd_test_pct <= X`

## Export paper trading

Dans l'UI, si un champion existe:
- bouton **Download paper manifest (JSON)**
- le JSON contient toutes les infos nécessaires pour reconstruire la config côté paper trading.

## Structure du code

- `streamlit_app.py`: UI principale
- `src/hyperliquid_ohlcv_optimizer/data/ohlcv_loader.py`: loader Parquet
- `src/hyperliquid_ohlcv_optimizer/strategies/`: stratégies + indicateurs
- `src/hyperliquid_ohlcv_optimizer/backtest/`: moteur de backtest (taker-only) + grid/martingale
- `src/hyperliquid_ohlcv_optimizer/optimize/optuna_runner.py`: logique d'analyse + rebuild report depuis `optuna.db`
- `src/hyperliquid_ohlcv_optimizer/optimize/optuna_worker.py`: worker (1 process = trials sur 1 stratégie)
- `src/hyperliquid_ohlcv_optimizer/optimize/run_optimize.py`: runner CLI (multiprocess SQLite + suivi terminal)
- `src/hyperliquid_ohlcv_optimizer/optimize/stop_run.py`: stop CLI (écrit `stop.flag`)

## Limites (v1)

- Exécution **taker-only** (pas de simulation maker/limit)
- OHLC-only: TP/SL intrabar approximés via `high/low`
- Funding/liquidations non modélisés
- Les résultats sont indicatifs: l'objectif est un ranking utile et exportable.
