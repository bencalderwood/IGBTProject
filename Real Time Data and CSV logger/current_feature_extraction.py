import numpy as np

def extract_current_features(ic, fs):

    ic = np.nan_to_num(ic)

    if len(ic) < 2:
        return {
            "Ic": 0,
            "ic_mean": 0,
            "ic_rms": 0,
            "ic_peak": 0,
            "ic_std": 0,
            "max_di_dt": 0
        }

    # Light smoothing to suppress switching noise
    window = min(5, len(ic))
    ic_smooth = np.convolve(
        ic,
        np.ones(window) / window,
        mode="same"
    )

    di_dt = np.gradient(ic_smooth) * fs

    return {
        "Ic": ic[-1],
        "ic_mean": np.mean(ic),
        "ic_rms": np.sqrt(np.mean(ic ** 2)),
        "ic_peak": np.max(np.abs(ic)),
        "ic_std": np.std(ic),
        "max_di_dt": np.max(np.abs(di_dt))
    }