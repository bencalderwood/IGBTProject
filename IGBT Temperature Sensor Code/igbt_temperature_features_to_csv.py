import pandas as pd
import os

def assign_temperature_health_label(T_mean):
    if T_mean < 75:
        return "Healthy"
    elif 75 <= T_mean <= 125:
        return "Warning"
    else:
        return "Fault"


def append_features_to_csv(features, label, filename):
    features["HealthState"] = label
    df = pd.DataFrame([features])

    if not os.path.exists(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)
