import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ==========================================
# 1. CHARGEMENT ET FUSION DES DONNÉES
# ==========================================

print("Chargement des fichiers CSV depuis le dossier /data/...")

path = './data/'
all_files = glob.glob(os.path.join(path, "5*.csv"))

if not all_files:
    print("ERREUR : Aucun fichier trouvé.")
    print("Vérifie que tes fichiers 501.csv, 502.csv, etc. sont dans le dossier 'data'.")
    exit()

li = []

for file in all_files:
    print("Lecture du fichier :", file)
    df = pd.read_csv(file)
    li.append(df)

data = pd.concat(li, axis=0, ignore_index=True)

print("Données chargées avec succès.")
print("Taille totale du dataset :", data.shape)


# ==========================================
# 2. NETTOYAGE DES DONNÉES
# ==========================================

if 'timestamp' in data.columns:
    data = data.drop('timestamp', axis=1)

data.dropna(inplace=True)

print("Taille après nettoyage :", data.shape)

# On prend seulement 100000 lignes pour accélérer l'entraînement
data_sampled = data.sample(n=100000, random_state=42)

print("Taille après échantillonnage :", data_sampled.shape)


# ==========================================
# 3. PRÉPARATION MATHÉMATIQUE DES DONNÉES
# ==========================================

X = data_sampled.drop('label', axis=1)
y = data_sampled['label']

# Transformation des labels en nombres
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

num_classes = len(np.unique(y_encoded))

print("Nombre de classes :", num_classes)
print("Classes détectées :", encoder.classes_)

# Séparation train/test : 80% entraînement, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Standardisation : z = (x - moyenne) / écart-type
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 4. MODÈLE CLASSIQUE 1 : RANDOM FOREST
# ==========================================

print("\nEntraînement du modèle classique 1 : Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)

y_pred_rf = rf_model.predict(X_test_scaled)

rf_acc = accuracy_score(y_test, y_pred_rf)

print("Random Forest terminé.")


# ==========================================
# 5. MODÈLE CLASSIQUE 2 : LOGISTIC REGRESSION
# ==========================================

print("\nEntraînement du modèle classique 2 : Logistic Regression...")

log_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)

log_model.fit(X_train_scaled, y_train)

y_pred_log = log_model.predict(X_test_scaled)

log_acc = accuracy_score(y_test, y_pred_log)

print("Logistic Regression terminé.")


# ==========================================
# 6. MODÈLE TINYML : PETIT RÉSEAU DE NEURONES
# ==========================================

print("\nEntraînement du modèle TinyML : Neural Network...")

tiny_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),

    tf.keras.layers.Dense(16, activation='relu'),

    tf.keras.layers.Dense(8, activation='relu'),

    tf.keras.layers.Dense(num_classes, activation='softmax')
])

tiny_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

tiny_model.fit(
    X_train_scaled,
    y_train,
    epochs=10,
    batch_size=32,
    verbose=1
)

_, tiny_acc = tiny_model.evaluate(X_test_scaled, y_test, verbose=0)

y_pred_tiny_prob = tiny_model.predict(X_test_scaled)

y_pred_tiny = np.argmax(y_pred_tiny_prob, axis=1)

print("TinyML Neural Network terminé.")


# ==========================================
# 7. CONVERSION DU MODÈLE TINYML EN TFLITE
# ==========================================

print("\nConversion du modèle TinyML en TensorFlow Lite...")

converter = tf.lite.TFLiteConverter.from_keras_model(tiny_model)
tflite_model = converter.convert()

with open("model_tiny.tflite", "wb") as f:
    f.write(tflite_model)

tiny_model_size_kb = len(tflite_model) / 1024

print("Modèle TinyML sauvegardé : model_tiny.tflite")
print(f"Taille du modèle TinyML : {tiny_model_size_kb:.2f} KB")


# ==========================================
# 8. AFFICHAGE DES RÉSULTATS GLOBAUX
# ==========================================

print("\n" + "=" * 55)
print("            COMPARAISON FINALE DES MODÈLES")
print("=" * 55)

print(f"Random Forest Accuracy        : {rf_acc:.4f}")
print(f"Logistic Regression Accuracy  : {log_acc:.4f}")
print(f"TinyML Neural Network Accuracy: {tiny_acc:.4f}")
print(f"Taille du modèle TinyML       : {tiny_model_size_kb:.2f} KB")

print("=" * 55)


# ==========================================
# 9. RAPPORTS DE CLASSIFICATION
# ==========================================

print("\nRapport de classification : Random Forest")
print(classification_report(y_test, y_pred_rf, zero_division=0))

print("\nRapport de classification : Logistic Regression")
print(classification_report(y_test, y_pred_log, zero_division=0))

print("\nRapport de classification : TinyML Neural Network")
print(classification_report(y_test, y_pred_tiny, zero_division=0))


# ==========================================
# 10. MATRICE DE CONFUSION RANDOM FOREST
# ==========================================

cm_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title("Matrice de Confusion : Random Forest")
plt.xlabel("Prédictions du modèle")
plt.ylabel("Vérité réelle")
plt.show()


# ==========================================
# 11. MATRICE DE CONFUSION LOGISTIC REGRESSION
# ==========================================

cm_log = confusion_matrix(y_test, y_pred_log)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Oranges')
plt.title("Matrice de Confusion : Logistic Regression")
plt.xlabel("Prédictions du modèle")
plt.ylabel("Vérité réelle")
plt.show()


# ==========================================
# 12. MATRICE DE CONFUSION TINYML
# ==========================================

cm_tiny = confusion_matrix(y_test, y_pred_tiny)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_tiny, annot=True, fmt='d', cmap='Greens')
plt.title("Matrice de Confusion : TinyML Neural Network")
plt.xlabel("Prédictions du modèle")
plt.ylabel("Vérité réelle")
plt.show()


# ==========================================
# 13. ANALYSE COMPARATIVE AUTOMATIQUE
# ==========================================

print("\n" + "=" * 55)
print("                ANALYSE COMPARATIVE")
print("=" * 55)

accuracies = {
    "Random Forest": rf_acc,
    "Logistic Regression": log_acc,
    "TinyML Neural Network": tiny_acc
}

best_model = max(accuracies, key=accuracies.get)

print("Meilleur modèle en accuracy :", best_model)
print(f"Accuracy du meilleur modèle : {accuracies[best_model]:.4f}")

print("\nAnalyse simple :")

if best_model == "Random Forest":
    print("Random Forest donne la meilleure précision globale.")
    print("Mais il est plus lourd qu'un petit réseau TinyML.")
elif best_model == "Logistic Regression":
    print("Logistic Regression donne la meilleure précision globale.")
    print("C'est un modèle simple, rapide et léger.")
else:
    print("TinyML donne la meilleure précision globale.")
    print("Il est aussi le plus adapté pour un système embarqué.")

print("\nPour un système IoT médical :")
print("- Random Forest est intéressant pour la précision.")
print("- Logistic Regression est simple et rapide.")
print("- TinyML est le plus adapté pour un appareil embarqué avec peu de mémoire.")
print("- Le choix final dépend du compromis entre accuracy, mémoire, vitesse et énergie.")

print("\n[INFO] Ferme les fenêtres des graphiques pour terminer le programme.")