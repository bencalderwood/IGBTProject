# ============================================================
# 1. Imports
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# 2. Load and Inspect Dataset
# ============================================================

df = pd.read_csv("temperature_RF_dataset.csv")

# Remove non-feature columns if present
df = df.drop(columns=["timestamp"], errors="ignore")

print("Dataset shape:", df.shape)
print(df.head())

# ============================================================
# 3. Feature / Label Separation
# ============================================================

X = df.drop(columns=["HealthState"])
y = df["HealthState"]

# Encode labels (Healthy / Warning / Fault)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Health classes:", label_encoder.classes_)

# ============================================================
# 4. Train / Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.30,
    stratify=y_encoded,
    random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ============================================================
# 5. Train Random Forest Classifier
# ============================================================

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# ============================================================
# 6. Model Evaluation
# ============================================================

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
)

print("\n================= MODEL PERFORMANCE =================")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", report)

# ---------------- Confusion Matrix ----------------

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

disp.plot(cmap="Blues", values_format="d")
plt.title("Temperature RF – Confusion Matrix")
plt.show()

# ============================================================
# 7. Feature Importance Analysis
# ============================================================

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n================= FEATURE IMPORTANCE =================")
print(importance_df)

plt.figure(figsize=(8, 5))
plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Temperature Feature Importance (Random Forest)")
plt.grid(True)
plt.show()

# ============================================================
# 8. Save Trained Model
# ============================================================

joblib.dump({
    "model": rf,
    "label_encoder": label_encoder,
    "feature_names": X.columns.tolist()
}, "temperature_rf_model.pkl")

print("\nModel saved as temperature_rf_model.pkl")


