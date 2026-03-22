import numpy as np

def extract_power_features(vce, ic):
    vce = np.nan_to_num(vce)
    ic = np.nan_to_num(ic)

    n = min(len(vce), len(ic))
    vce = vce[:n]
    ic = ic[:n]

    if n == 0:
        return {
            "p_inst": 0,
            "p_mean": 0,
            "p_peak": 0,
            "p_std": 0
        }

    # Instantaneous power (stress proxy)
    p_inst = np.abs(vce * ic)

    return {
        "p_instantaneous": p_inst[-1],
        "p_mean": np.mean(p_inst),
        "p_peak": np.max(p_inst),
        "p_std": np.std(p_inst)
    }