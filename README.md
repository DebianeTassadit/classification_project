# Classification Project

## Description

Ce projet est une simulation de projet d'entreprise dans laquelle nous collaborons pour créer une solution complète de génération de données synthétiques, d'entraînement et d'évaluation de modèles de classification. Il inclut également l'intégration des scripts dans un conteneur Docker pour faciliter l'exécution et le déploiement.

---

## Objectifs

- Générer des données synthétiques et entraîner différents modèles de classification (`Random Forest`, `SVM`, `Logistic Regression`).
- Évaluer les performances des modèles sur des données de test et de validation.
- Intégrer les fonctionnalités développées dans un pipeline complet (au niveau de Github et Docker).
- Dockeriser le projet pour simplifier son exécution et son déploiement.

---

## Structure du Projet
- **`Data_Generate`** : Notebook pour analyser et visualiser diverses données tout en explorant les variations des paramètres.
- **`train_classifier.py`** : Script pour générer des données a partir des  paramètres sélectionnés du notebook, entraîner les modèles, et exporter le modèle final ainsi que les données de validation.
- **`predict_classification.py`** : Script pour importer le modèle et les données de validation, effectuer des prédictions, et évaluer les performances finales.
- **`requirements.txt`** : Liste des dépendances Python nécessaires pour exécuter le projet.
- **Docker Integration** :
  - **`Dockerfile`** : Fichier pour construire un conteneur exécutant les scripts.
  - **`docker-compose.yml`** : Configuration pour gérer les conteneurs.

---

## Fonctionnalités

### Génération des Données :
- Utilisation de `make_classification` de scikit-learn pour créer des données synthétiques.
- Paramètres ajustables comme `n_informative`, `n_classes`, `class_sep`. `n_clusters_per_class`

### Entraînement des Modèles :
- Implémentation de plusieurs algorithmes de classification :
  - Random Forest
  - SVM
  - Régression Logistique
- Cross validation pour évaluer les performances sur des données d'entraînement.

### Évaluation des Modèles :
- Calcul des métriques : précision, rappel, F1-score et rapport de classification.
- Export des résultats et des modèles.

### Prédiction :
- Importation du modèle et des données de validation pour effectuer des prédictions.
- Évaluation finale des résultats.

### Dockerisation :
- Automatisation de l'entraînement et de la prédiction dans un conteneur Docker.
- Configuration avec Docker Compose pour une exécution simplifiée.

---


# Utilisation du Pipeline Docker pour l'Entraînement et la Prédiction

## Instructions

### Construire les images Docker
```bash
docker-compose build
```

### Exécuter le script d'entraînement
```bash
docker-compose run training
```

### Exécuter le script de prédiction
```bash
docker-compose run prediction
```

---

## Interprétation des Résultats

### Résultats de l'entraînement (`train_classifier.py [logistic regression]`)
Vous avez aussi le choix de prendre d'autres modèles

#### Modèle : Logistic Regression
- **Scores de Cross Validation** : `[0.9921, 0.9914, 0.9928, 0.9928, 0.9921]`
  - Ces résultats montrent une stabilité et une cohérence élevées des performances sur les différents splits des données.
- **Précision globale sur les données de test** : `99%`
  - Cela signifie que le modèle a correctement classifié 99% des échantillons de test.

#### Rapport de classification
- Les métriques telles que la précision, le rappel et le F1-score pour chaque classe sont presque parfaites (supérieures à `0.99`), indiquant un excellent équilibre dans la performance sur toutes les classes.

---

### Résultats des prédictions (`predict_classification.py`)

#### Évaluation sur les données de validation
- **Précision globale** : `99%`
  - Le modèle conserve sa performance élevée sur des données qu'il n'a jamais vues auparavant.
- **Rapport de classification** : 
  - Chaque classe obtient des scores presque parfaits en précision, rappel et F1-score.

---
