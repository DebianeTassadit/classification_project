# Étape 1 : Utiliser une image Python légère comme base
FROM python:3.9-slim

# Étape 2 : Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Étape 3 : Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt .

# Étape 4 : Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Étape 5 : Copier les scripts dans le conteneur
COPY train_classifier.py predict_classification.py . 

# Étape 6 : Définir la commande par défaut pour lancer l’entraînement et la prédiction
CMD ["sh", "-c", "python train_classifier.py && python predict_classification.py"]
