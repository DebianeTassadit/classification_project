import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
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

# Étape 3 : Réduction de dimensions avec PCA
def apply_pca(X_train, X_test, X_val, n_components=2):
    """
    Réduit les dimensions des données à 2 dimensions pour la visualisation.
    """
    print(f"Réduction des dimensions à {n_components} avec PCA...")
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    X_val_pca = pca.transform(X_val)
    return X_train_pca, X_test_pca, X_val_pca, pca

# Étape 4 : Entraînement avec cross-validation
def cross_validate_model(X_train, y_train, model):
    """
    Effectue une cross-validation sur les données d'entraînement et retourne la moyenne des scores.
    """
    print("Cross-validation en cours...")
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Scores de cross-validation : {scores}")
    print(f"Accuracy moyenne : {scores.mean():.2f}")
    return model

# Étape 5 : Entraînement du modèle
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

# Étape 6 : Évaluation des performances
def evaluate_model(model, X_test, y_test):
    """
    Évalue les performances du modèle sur les données de test.
    """
    print("Évaluation des performances sur les données de test...")
    y_pred = model.predict(X_test)
    print("Rapport de classification :")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.2f}")

# Étape 7 : Exportation des fichiers
def export_files(model, X_val, y_val):
    """
    Exporte le modèle entraîné et les données de validation.
    """
    print("Exportation du modèle et des données de validation...")
    joblib.dump(model, MODEL_PATH)
    val_data = pd.DataFrame(X_val, columns=["PC1", "PC2"])
    val_data["Target"] = y_val.reset_index(drop=True)
    val_data.to_csv(VALIDATION_DATA_PATH, index=False)
    print(f"Modèle sauvegardé dans {MODEL_PATH}")
    print(f"Données de validation sauvegardées dans {VALIDATION_DATA_PATH}")

# Étape 8 : Visualisation avec frontière de décision
def plot_decision_boundary(model, X, y, title="Frontière de décision"):
    """
    Visualise les données et la frontière de décision pour un modèle entraîné.
    """
    # Création d'une grille pour tracer les frontières
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Prédiction sur chaque point de la grille
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Tracer la frontière de décision
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap="viridis")

    # Tracer les points de données
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap="viridis")
    plt.title(title)
    plt.xlabel("Composante principale 1")
    plt.ylabel("Composante principale 2")
    plt.colorbar(scatter, label="Classes")
    plt.show()

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

    # Réduction des dimensions (facultatif pour visualisation)
    #X_train, X_test, X_val, pca = apply_pca(X_train, X_test, X_val)

    # Entraînement du modèle
    model = train_model(X_train, y_train, model_choice)

    # Évaluation des performances
    evaluate_model(model, X_test, y_test)

    # Exportation du modèle et des données de validation
    export_files(model, X_val, y_val)

    # Visualisation avec la frontière de décision
    print("Visualisation avec la frontière de décision...[optionel] => enlever les commentaires du code")
    #plot_decision_boundary(model, X_train, y_train, title=f"Frontière de décision {model_choice} (train)")

    print("Pipeline terminé avec succès.")
