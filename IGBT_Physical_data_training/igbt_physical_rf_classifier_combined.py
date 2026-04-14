import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load combined dataset
df = pd.read_csv("IGBT_healthy_warning_combined.csv")

df["FaultType"] = df["FaultType"].fillna("None")
# =========================
# Separate features and targets
# =========================
X = df.drop(columns=["HealthState", "FaultType"])

y_health = df["HealthState"]
y_fault = df["FaultType"]

# =========================
# Encode HealthState
# =========================
le_health = LabelEncoder()
y_health_encoded = le_health.fit_transform(y_health)

health_map = dict(zip(le_health.classes_, le_health.transform(le_health.classes_)))

# =========================
# Encode FaultType
# =========================
le_fault = LabelEncoder()
y_fault_encoded = le_fault.fit_transform(y_fault)

fault_map = dict(zip(le_fault.classes_, le_fault.transform(le_fault.classes_)))

# =========================
# Combine targets
# =========================
#y_encoded = np.column_stack((y_health_encoded, y_fault_encoded))

# =========================
# Print mappings
# =========================
print("HealthState encoding:", health_map)
print("FaultType encoding:", fault_map)


# Train/test split
X_train, X_test, y_health_train, y_health_test, y_fault_train, y_fault_test = train_test_split(
    X,
    y_health_encoded,
    y_fault_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_health_encoded   # ONLY stratify on main label
)

# Train RF on HealthState Label
rf_health = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)
rf_health.fit(X_train, y_health_train)
y_health_pred = rf_health.predict(X_test)

# Train RF on FaultType Label
rf_fault = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)
rf_fault.fit(X_train, y_fault_train)
y_fault_pred = rf_fault.predict(X_test)

# Evaluation - HealthState
# =========================
print("\n=== HealthState Model ===")
print("Accuracy:", accuracy_score(y_health_test, y_health_pred))
print(classification_report(y_health_test, y_health_pred, target_names=le_health.classes_))

cm_health = confusion_matrix(y_health_test, y_health_pred)
disp = ConfusionMatrixDisplay(cm_health, display_labels=le_health.classes_)
disp.plot()
plt.title("HealthState Confusion Matrix")
plt.show()

# =========================
# Evaluation - FaultType
# =========================
print("\n=== FaultType Model ===")
print("Accuracy:", accuracy_score(y_fault_test, y_fault_pred))
print(classification_report(
    y_fault_test,
    y_fault_pred,
    labels=range(len(le_fault.classes_)),   # force all labels
    target_names=le_fault.classes_
))

cm_fault = confusion_matrix(y_fault_test, y_fault_pred, labels=range(len(le_fault.classes_)))
disp = ConfusionMatrixDisplay(cm_fault, display_labels=le_fault.classes_)
disp.plot()
plt.title("FaultState Confusion Matrix")
plt.show()


# Feature importance

importance_health = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_health.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n================= FEATURE IMPORTANCE =================")
print(importance_health)

importance_fault = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_fault.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(
    importance_health["Feature"],
    importance_health["Importance"]
)

plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Feature Importance (Random Forest)")
plt.grid(True)
plt.savefig("igbt_rf_Healthstate_feature_importance.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
plt.barh(
    importance_fault["Feature"],
    importance_fault["Importance"]
)

plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Feature Importance (Random Forest)")
plt.grid(True)
plt.savefig("igbt_rf_Faultstate_feature_importance.png", dpi=300)
plt.show()