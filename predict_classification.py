import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score

# Chemins des fichiers importés
MODEL_PATH = "model.pkl"  # Chemin du modèle
VALIDATION_DATA_PATH = "validation_data.csv"  # Chemin des données de validation

def load_model_and_data(model_path, data_path):
    """
    Charge le modèle entraîné et les données de validation.
    """
    print("Chargement du modèle et des données de validation...")

    # Charger le modèle
    model = joblib.load(model_path)
    print("Modèle chargé avec succès.")

    # Charger les données de validation
    validation_data = pd.read_csv(data_path)
    print("Données de validation chargées avec succès.")

    # Séparer les caractéristiques et la cible
    X_val = validation_data.drop(columns=["Target"])
    y_val = validation_data["Target"]

    return model, X_val, y_val

def predict_and_evaluate(model, X_val, y_val):
    """
    Effectue des prédictions sur les données de validation et évalue les performances.
    """
    print("Prédictions en cours...")
    y_pred = model.predict(X_val)

    print("Évaluation des performances...")
    print("Rapport de classification :")
    print(classification_report(y_val, y_pred))
    print(f"Accuracy : {accuracy_score(y_val, y_pred):.2f}")

def main():
    print("Bienvenue dans le script de prédiction et d'évaluation !")

    # Charger le modèle et les données
    model, X_val, y_val = load_model_and_data(MODEL_PATH, VALIDATION_DATA_PATH)

    # Effectuer des prédictions et évaluer
    predict_and_evaluate(model, X_val, y_val)

    print("Processus terminé avec succès.")

if __name__ == "__main__":
    main()

"""Interpretation: Notre modèle montre une excellente performance sur les données de validation,
 représentant des données inédites. Avec une précision d'environ 98% tant pendant 
 l'entraînement que lors des prédictions, nous pouvons conclure que le modèle ne souffre 
 ni de surajustement (overfitting) ni de sous-ajustement (underfitting).
 Cette stabilité suggère une bonne généralisation du modèle, 
capable de prédire efficacement des données qu'il n'a jamais vues auparavant."""