# Initialiser / cloner un dépôt
## Créer un nouveau repo local
git init

## Cloner un repo distant
git clone https://github.com/ton-compte/ton-projet.git
cd ton-projet

# Créer et gérer des branches
## Voir toutes les branches
git branch

## Créer une nouvelle branche et y basculer
git checkout -b nom-branch

## Passer d’une branche à l’autre
git checkout nom-branch

## Supprimer une branche locale
git branch -d nom-branch

# Ajouter / supprimer / déplacer des fichiers
## Ajouter un fichier au suivi Git
git add sql/kpis/kpi_revenue_monthly.sql

## Ajouter tous les fichiers modifiés
git add .

## Supprimer un fichier
git rm chemin_du_fichier

## Renommer / déplacer un fichier
git mv sql/kpis/ancien_kpi.sql sql/kpis/nouveau_kpi.sql

# Commits
## Créer un commit avec message
git commit -m "feat(kpi): ajout KPI Monthly Active Users"

## Modifier le dernier commit (si erreur de message ou oublie d’un fichier)
git commit --amend

## Ajouter un fichier oublié au dernier commit
git add sql/kpis/kpi_churn_rate.sql
git commit --amend --no-edit

# Pousser / récupérer des modifications
## Pousser la branche courante vers le remote
git push origin feature/kpi-mau

## Pousser la branche vers remote et définir l’upstream
git push -u origin feature/kpi-mau

## Récupérer les dernières modifications du remote
git pull origin main

## Récupérer toutes les branches et updates
git fetch --all

# Visualiser l’historique et les changements
## Voir l’état des fichiers
git status

## Voir les différences entre modifications et dernier commit
git diff

## Voir l’historique des commits
git log --oneline --graph --decorate --all

## Voir les fichiers modifiés dans le dernier commit
git show --name-only

# Travailler avec les Pull Requests (PR)
## Créer une branche pour une PR
git checkout -b kpi/kpi-revenue

## Ajouter, commit, push
git add .
git commit -m "feat(kpi): ajout KPI Revenue"
git push -u origin feature/kpi-revenue

## Aller sur GitHub → ouvrir une Pull Request depuis cette branche

# Rebaser / fusionner
## Fusionner main dans votre branche feature
git checkout feature/kpi-mau
git pull origin main --rebase

## Fusionner votre branche feature dans main (local)
git checkout main
git merge feature/kpi-mau

# Annuler / restaurer
## Annuler toutes les modifications d’un fichier
git restore sql/kpis/kpi_churn_rate.sql

## Annuler un fichier déjà ajouté (unstage)
git restore --staged sql/kpis/kpi_churn_rate.sql

## Revenir à un commit précédent (danger : perte de modifications locales)
git reset --hard <commit_hash>
