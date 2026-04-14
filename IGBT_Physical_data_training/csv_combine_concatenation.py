import pandas as pd

# Load CSVs
bond_wire_fatigue_df = pd.read_csv("igbt_features_warning_bond-wire_fatigue_temp.csv")
gate_oxide_degradation_df = pd.read_csv("igbt_features_warning_temp_gate_oxide_degradation.csv")

# Keep HealthState from one file only
labels = bond_wire_fatigue_df["HealthState"]

# Drop duplicate label column from second dataset
gate_oxide_df = gate_oxide_degradation_df.drop(columns=["HealthState"])

# Combine side-by-side
merged_df = pd.concat([bond_wire_fatigue_df.drop(columns=["HealthState"]), gate_oxide_degradation_df], axis=0)

# Add label back
merged_df["HealthState"] = labels

# Save
merged_df.to_csv("IGBT_physical_warning_merged.csv", index=False)

print("✅ Dataset merged correctly using concatenation")