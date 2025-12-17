# Roadmap — ohlcv-optimizer-v1

Ce document liste les priorités et chantiers prévus pour stabiliser/améliorer l’optimiseur OHLCV (v1), l’UI Streamlit et le workflow runs/report.

## Objectifs produit

- Avoir un pipeline **Optimize → Analyze → Export** fiable (reproductible, traçable, explicable).
- Rendre les métriques et décisions de sélection **auditables** (pourquoi ce champion ? quels paramètres ?).
- Permettre une itération rapide sur les stratégies et la méthode d’optimisation (PM, TP/SL, risk).

## État actuel (résumé)

- UI Streamlit : `Optimize`, `Backtest`, `Analyze`.
- Optimisation Optuna multi-objectif sur TRAIN (métrique configurable) + drawdown.
- Paramètres méthode d’optimisation : PM (`none|grid|martingale`), RR forcé, objective metric.
- Runs persistés : `runs/<run_id>...` avec `context.json`, `optuna.db`, `status.json`, `progress.jsonl`, `report.json`.

## Priorités court-terme (P0)

### P0 — Fiabilité workflow runs/report
- **Progress bar** pendant `Force rebuild report from optuna.db` (feedback utilisateur, long running).
- Vérification systématique des invariants run:
  - `storage_url` local par défaut (éviter réutilisation involontaire)
  - `study_name_prefix` cohérent
  - `report.json` non vide si run terminé
- Hardening Analyze:
  - affichage des périodes (opt/train/test)
  - fallback robustes (runs anciens, champs manquants)

### P0 — Compatibilité “RR forcé”
- Revue de compatibilité:
  - RR forcé + `grid`/`martingale`
  - RR forcé + `sl_trailing`
  - RR forcé + `tp_mgmt=partial_trailing`
- Définir clairement les règles (ex: TP ne suit pas un SL trailing) et les valider sur cas limites.

### P0 — Run de validation
- Lancer une run “clean” (nouveau `run_dir`, `optuna.db` local) et vérifier:
  - `optuna.db` créé
  - `report.json` produit
  - Analyze charge sans rebuild

## Améliorations court/moyen terme (P1)

### P1 — UX/Analyse
- Génération/stockage des **candidate analytics** (et bouton “Compute analytics now” si absent).
- Optimisation UX Analyze:
  - caching des CSV/JSON (Streamlit `cache_data`)
  - “download bundle” (candidate+params+analytics) pour debug/partage

### P1 — Tests et robustesse
- Tests unitaires sur:
  - build config (`OptimizationConfig` → `BacktestConfig`)
  - règles RR forcé
  - sizing `risk` vs `fixed_notional`
- Golden tests: sample run minimal + vérification artefacts (`report.json` schema minimal).

### P1 — Refactor architecture
- Sortir progressivement le code Analyze de `streamlit_app.py`:
  - `ui/analyze.py` (render)
  - `analysis/loaders.py` (I/O)
  - `analysis/compute.py` (pur, testable)

## Évolutions moyen/long terme (P2)

### P2 — Optimisation avancée
- Walk-forward / rolling windows (plusieurs splits) + agrégation scoring.
- Contraintes plus riches dans l’objectif (ex: min trades, min PF, max liquidation rate).
- Modes de sélection champion alternatifs (stabilité, robustesse multi-splits).

### P2 — Backtest realism
- Modélisation funding/fees plus fine (perps), slippage plus réaliste.
- Maker/limit simulation (optionnelle).
- Event model plus strict (fills intrabar, partial fills).

### P2 — Intégration exécution/paper
- Normaliser le format “manifest” (versioning, schema).
- Pipeline paper trading (hors infra) + import/export stable.
- Sources supplémentaires (cTrader/Open API) si reprise.

## Dette technique / housekeeping

- Clarifier le schéma `report.json` (versioning + champs requis).
- Standardiser la nomenclature des métriques (train/test) et la documentation.
- Réduire les duplications de clés metrics dans l’UI (tables de mapping).

## Notes

- Cette roadmap est volontairement orientée “v1”: l’objectif principal est la fiabilité et l’itération rapide.
