import pandas as pd

# =========================
# Load CSV files
# =========================
bond_wire_fatigue_df = pd.read_csv("igbt_features_warning_bond-wire_fatigue_temp.csv")
gate_oxide_degradation_df = pd.read_csv("igbt_features_warning_temp_gate_oxide_degradation.csv")


#dynamic_df = dynamic_df.dropna()

# Option 2: fill with median
#dynamic_df["Vge_plateau"] = dynamic_df["Vge_plateau"].fillna(dynamic_df["Vge_plateau"].median())

# =========================
# Define merge keys
# (adjust if you have more)
# =========================
#merge_keys = ["Gate_voltage", "Temperature"]

# =========================
# Preserve ONE HealthState
# =========================
health_thermal = bond_wire_fatigue_df[merge_keys + ["HealthState"]].copy()

# Drop HealthState from others to avoid duplication
bond_wire_fatigue_df = bond_wire_fatigue_df.drop(columns=["HealthState"])
gate_oxide_degradation_df = gate_oxide_degradation_df.drop(columns=["HealthState"])
#thermal_df = thermal_df.drop(columns=["HealthState"])

# =========================
# Merge datasets
# =========================
merged_df = bond_wire_fatigue_df.merge(
    gate_oxide_degradation_df,
    on=merge_keys,
    how="inner"
)

merged_df = merged_df.merge(
    bond_wire_fatigue_df,
    on=merge_keys,
    how="inner"
)

# =========================
# Reattach HealthState
# =========================
merged_df = merged_df.merge(
    health_thermal,
    on=merge_keys,
    how="inner"
)

# =========================
# Sanity checks
# =========================
print("Merged shape:", merged_df.shape)
print("Missing values per column:")
print(merged_df.isna().sum())

# Check label consistency
label_counts = merged_df["HealthState"].value_counts()
print("\nHealthState distribution:")
print(label_counts)

# =========================
# Save clean dataset
# =========================
merged_df.to_csv("IGBT_physical_warning_merged.csv", index=False)

print("\n✅ Clean ML-ready dataset saved as IGBT_merged_clean.csv")