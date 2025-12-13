# quantlab

Monorepo pour des projets de **quant/algo trading**.

L’objectif est de garder :

- un repo unique (vision globale)
- des projets isolés (dépendances / versions Python / langages séparés)
- un workflow Git simple et propre (branches + PR)
- **aucun secret** dans Git

## Structure du repo

- `projects/`
  - Projets “exécutables” (scripts, apps, services). Chaque projet a son propre environnement et ses dépendances.
- `libs/`
  - Code partagé réutilisable entre projets (packages internes). Optionnel au début.
- `research/`
  - Expérimentations, notebooks, brouillons. Rien d’ici ne doit être requis pour exécuter les projets “propres”.
- `infra/` (optionnel)
  - Docker, compose, déploiement, monitoring.

### Projets actuels

- `projects/auto-optimizer-v1/`
  - Version 1 d’un “auto optimizer” (actuellement en mode script).

### Projets futurs (exemples)

- `projects/auto-optimizer-v2/`
- `projects/ml-*/` (modèles ML, research-to-prod, etc.)
- `projects/dashboard/` (UI)

## Principes d’organisation (à respecter)

- **Un projet = un dossier** dans `projects/`.
- **Isolation totale** : chaque projet gère ses dépendances (et peut avoir une version Python différente).
- Le code partagé doit aller dans `libs/` (au lieu de copier-coller dans chaque projet).
- Les expérimentations vont dans `research/` pour éviter de polluer les projets.

## Démarrage rapide (Windows)

Chaque projet est autonome. Exemple (à adapter au projet) :

1) Ouvrir un terminal dans le dossier du projet

2) Créer un environnement virtuel

```bash
py -m venv .venv
.venv\Scripts\activate
```

3) Installer les dépendances

```bash
pip install -r requirements.txt
```

4) Lancer le script

```bash
python src\...\main.py
```

> Chaque projet aura ses instructions exactes dans son propre `README.md`.

## Workflow Git (simple et professionnel)

- **Ne pas travailler sur `main`**.
- **1 tâche = 1 branche** :
  - `feature/<sujet>` (nouvelle fonctionnalité)
  - `fix/<sujet>` (bugfix)
  - `experiment/<sujet>` (expérience courte)
- Ouvrir une **Pull Request** (PR) et fusionner en **Squash**.
- `main` doit rester **stable**.

### Routine conseillée

```bash
git checkout main
git pull
git checkout -b feature/mon-sujet
# ... changements ...
git add .
git commit -m "Describe change"
git push -u origin feature/mon-sujet
```

Puis : PR sur GitHub → **Squash and merge** → suppression de la branche.

## Politique sécurité (secrets / données)

### Interdits dans Git

- clés API (Binance, Bybit, etc.)
- tokens, mots de passe
- fichiers `.env`
- credentials cloud
- dumps de base de données
- datasets lourds ou sensibles

### À faire à la place

- Utiliser des variables d’environnement via un fichier local **non commit** : `.env`
- Mettre un exemple safe dans `.env.example`
- Stocker les secrets dans :
  - GitHub Secrets (CI)
  - un gestionnaire de secrets (si infra plus tard)

### Vérification rapide avant push

- vérifier `git status`
- vérifier les fichiers ajoutés dans le diff
- s’assurer qu’aucun secret n’apparaît

## Versioning

Dans un monorepo, on tagge avec un préfixe par projet (quand nécessaire) :

- `auto-optimizer-v1@v0.1.0`
- `auto-optimizer-v2@v0.1.0`
- `quantlab-core@v0.1.0`

## Licence

À définir.
