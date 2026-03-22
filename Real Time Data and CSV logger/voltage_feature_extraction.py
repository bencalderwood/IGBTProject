import numpy as np

def extract_vce_features(vce, fs):
    vce = np.nan_to_num(vce) # NaN readings to number

    if len(vce) < 2:
        return {
            "vce": 0,
            "vce_mean": 0,
            "vce_max": 0,
            "vce_min": 0,
            "vce_std": 0,
            "max_dv_dt": 0,
            "vce_overshoot": 0
        }

    # Light smoothing to suppress switching noise
    window = min(5, len(vce))
    vce_smooth = np.convolve(
        vce,
        np.ones(window)/window,
        mode="same"
    )

    dv_dt = np.gradient(vce) * fs

    vce_mean = np.mean(vce)

    return {
        "vce": vce[-1],
        "vce_mean": vce_mean, # Average operating voltage
        "vce_max": np.max(vce), # Stress extremes
        "vce_min": np.min(vce),
        "vce_std": np.std(vce), # Ripple / noise
        "max_dv_dt": np.max(np.abs(dv_dt)), # Switching stress
        "vce_overshoot": np.max(vce) - vce_mean # Transient overshoot detection
    }