import pandas as pd

# Load CSVs
healthy_dcsupply_2_5v_df = pd.read_csv("igbt_features_healthy_nofault_temp_accel.csv")
healthy_dcsupply_5v_df = pd.read_csv("igbt_features_healthy_temp_accel_no_fault.csv")

# Keep HealthState from one file only
labels = healthy_dcsupply_2_5v_df["HealthState"]

# Drop duplicate label column from second dataset
healthy_dcsupply_5v_df = healthy_dcsupply_5v_df.drop(columns=["HealthState"])

# Combine side-by-side
merged_df = pd.concat([healthy_dcsupply_2_5v_df.drop(columns=["HealthState"]), healthy_dcsupply_5v_df], axis=0)

# Add label back
merged_df["HealthState"] = labels

# Save
merged_df.to_csv("IGBT_physical_healthy_temp_accel_merged.csv", index=False)

print("✅ Dataset merged correctly using concatenation")