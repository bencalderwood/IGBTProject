import pandas as pd

healthy1 = pd.read_csv("igbt_features_healthy_nofault_temp_accel.csv")
healthy2 = pd.read_csv("igbt_features_healthy_temp_accel_no_fault.csv")
healthy3 = pd.read_csv("igbt_features_healthy_temp_no_fault3.csv")

warning = pd.read_csv("IGBT_physical_warning_merged.csv")
#fault = pd.read_csv("your_fault_dataset.csv")

healthy_df = pd.concat([healthy1, healthy2, healthy3], ignore_index=True)

common_cols = list(
    set(healthy_df.columns) &
    set(warning.columns) #&
    #set(fault.columns)
)

healthy_df = healthy_df[common_cols]
warning = warning[common_cols]
#fault = fault[common_cols]

df = pd.concat([healthy_df, warning], ignore_index=True)

# SAVE TO CSV
df.to_csv("IGBT_healthy_warning_combined.csv", index=False)

print("✅ Combined dataset saved successfully")