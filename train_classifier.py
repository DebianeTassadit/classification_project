import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Chemins pour l'exportation des fichiers
MODEL_PATH = "model.pkl"
VALIDATION_DATA_PATH = "validation_data.csv"

# Étape 1 : Génération des données synthétiques
def generate_data(n_samples=10000, n_features=10, n_informative=5, n_classes=4,
                  n_clusters_per_class=1, class_sep=3.0):
    """
    Génère des données synthétiques avec les paramètres choisis.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class,
        class_sep=class_sep,
        random_state=42
    )
    return pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(X.shape[1])]), pd.Series(y, name="Target")

# Étape 2 : Division des données
def split_data(data, target):
    """
    Divise les données en ensembles d'entraînement, de test et de validation.
    """
    print("Division des données en ensembles d'entraînement, de test et de validation...")
    X_train, X_temp, y_train, y_temp = train_test_split(data, target, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)
    return X_train, X_test, X_val, y_train, y_test, y_val

# Étape 3 : Entraînement avec cross-validation
def cross_validate_model(X_train, y_train, model):
    """
    Effectue une cross-validation sur les données d'entraînement et retourne la moyenne des scores.
    """
    print("Cross-validation en cours...")
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Scores de cross-validation : {scores}")
    print(f"Accuracy moyenne : {scores.mean():.2f}")
    return model

# Étape 4 : Entraînement du modèle
def train_model(X_train, y_train, model_choice):
    """
    Entraîne un modèle choisi par l'utilisateur.
    """
    print(f"Entraînement du modèle {model_choice}...")
    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_choice == "SVM":
        model = SVC(kernel="rbf", gamma="scale", C=1.0, random_state=42)
    elif model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        raise ValueError("Choix du modèle non valide.")

    # Cross-validation avant entraînement final
    model = cross_validate_model(X_train, y_train, model)

    # Entraînement final sur toutes les données d'entraînement
    model.fit(X_train, y_train)
    return model

# Étape 5 : Évaluation des performances
def evaluate_model(model, X_test, y_test):
    """
    Évalue les performances du modèle sur les données de test.
    """
    print("Évaluation des performances sur les données de test...")
    y_pred = model.predict(X_test)
    print("Rapport de classification :")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.2f}")

# Étape 6 : Exportation des fichiers
def export_files(model, X_val, y_val):
    """
    Exporte le modèle entraîné et les données de validation.
    """
    print("Exportation du modèle et des données de validation...")

    # Vérification que X_val et y_val ont la même taille
    assert len(X_val) == len(y_val), "La taille de X_val et y_val ne correspond pas. Vérifiez la division des données."

    # Alignement explicite entre X_val et y_val
    y_val = y_val.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)

    # Ajout de la colonne Target
    val_data = X_val.copy()
    val_data["Target"] = y_val

    # Sauvegarde
    joblib.dump(model, MODEL_PATH)
    val_data.to_csv(VALIDATION_DATA_PATH, index=False)

    print(f"Modèle sauvegardé dans {MODEL_PATH}")
    print(f"Données de validation sauvegardées dans {VALIDATION_DATA_PATH}")

# Main
if __name__ == "__main__":
    print("Bienvenue dans le pipeline d'entraînement de modèle !")
    print("Veuillez choisir un modèle parmi les options suivantes :")
    print("1 - Random Forest")
    print("2 - SVM")
    print("3 - Logistic Regression")
    
    choice = input("Entrez le numéro du modèle que vous souhaitez utiliser : ")
    
    model_mapping = {
        "1": "Random Forest",
        "2": "SVM",
        "3": "Logistic Regression"
    }
    
    model_choice = model_mapping.get(choice)
    if not model_choice:
        print("Choix invalide. Veuillez relancer le script et choisir une option valide.")
        exit()

    print(f"Vous avez choisi : {model_choice}")

    # Génération des données synthétiques
    print("Génération des données synthétiques...")
    X, y = generate_data()
    print("Données générées avec succès.")

    # Division des données
    X_train, X_test, X_val, y_train, y_test, y_val = split_data(X, y)

    # Entraînement du modèle
    model = train_model(X_train, y_train, model_choice)

    # Évaluation des performances
    evaluate_model(model, X_test, y_test)

    # Exportation du modèle et des données de validation
    export_files(model, X_val, y_val)

    print("Pipeline terminé avec succès.")
