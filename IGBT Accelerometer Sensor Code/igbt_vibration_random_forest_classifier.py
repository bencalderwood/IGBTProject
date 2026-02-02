# ============================================================
# 1. Imports
# ============================================================
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

# ============================================================
# 2. Load Dataset
# ============================================================

# Load vibration dataset
df = pd.read_csv("vibration_RF_dataset.csv")

# Drop timestamp (not useful for RF)
df = df.drop(columns=["timestamp"], errors="ignore")

print(df.head())
print(df.info())

# ============================================================
# 3. Prepare Features and Labels
# ============================================================

label_col = "HealthState"

X = df.drop(columns=[label_col])
y = df[label_col]

# Encode labels: Healthy=0, Warning=1, Fault=2
encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)

# ============================================================
# 4. Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_enc,
    test_size=0.25,
    random_state=42,
    stratify=y_enc
)

# ============================================================
# 5. Train Random Forest
# ============================================================

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# ============================================================
# 6. Evaluation
# ============================================================

y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:", classification_report(y_test,y_pred,target_names=encoder.classes_))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
)

disp.plot(cmap="Blues")
plt.title("Vibration RF Confusion Matrix")
plt.show()

# ============================================================
# 7. Feature Importance
# ============================================================

importances = rf.feature_importances_
feature_names = X.columns

feat_imp = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print(feat_imp)

plt.figure(figsize=(8,5))
plt.barh(feat_imp["Feature"], feat_imp["Importance"])
plt.gca().invert_yaxis()
plt.title("Random Forest Feature Importance (Vibration)")
plt.xlabel("Importance")
plt.show()



