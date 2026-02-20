import numpy as np

def extract_temperature_features(temp, fs):

    if len(temp) < 2:
        return {
            "temp_mean": 0,
            "temp_max": 0,
            "temp_min": 0,
            "temp_std": 0,
            "temp_rise_rate": 0,
            "temp_peak_gradient": 0
        }

        # Light smoothing to reduce noise
    window = min(5, len(temp))
    temp_smooth = np.convolve(
        temp,
        np.ones(window) / window,
        mode="same"
    )

    dT_dt = np.gradient(temp_smooth) * fs

    return {
        "temp_mean": np.mean(temp_smooth),
        "temp_max": np.max(temp_smooth),
        "temp_min": np.min(temp_smooth),
        "temp_std": np.std(temp_smooth),
        "temp_rise_rate": np.mean(dT_dt),
        "temp_peak_gradient": np.max(np.abs(dT_dt))
    }