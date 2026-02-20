import numpy as np
from scipy.fft import rfft, rfftfreq

def magnitude(ax, ay, az):
    return np.sqrt(ax**2 + ay**2 + az**2)

def extract_vibration_features(ax, ay, az):
    #Handling NaNs (Not a Number)
    ax = np.nan_to_num(ax)
    ay = np.nan_to_num(ay)
    az = np.nan_to_num(az)

    n = min(len(ax), len(ay), len(az))
    if n == 0:
        return {
            "vib_rms": 0,
            "vib_peak": 0,
            "vib_std": 0,
            "vib_kurtosis": 0,
            "vib_crest_factor": 0
        }

    mag = np.sqrt(ax[:n]**2 + ay[:n]**2 + az[:n]**2)
    mag -= np.mean(mag)

    rms = np.sqrt(np.mean(mag**2))
    std = np.std(mag) + 1e-8

    return {
        "vib_rms": rms,
        "vib_peak": np.max(np.abs(mag)),
        "vib_std": std,
        "vib_kurtosis": np.mean(mag ** 4) / (std ** 4),
        "vib_crest_factor": np.max(np.abs(mag)) / (rms + 1e-8)
    }

def extract_fft_features(signal, fs):
    signal = np.nan_to_num(signal) # NaN handling

    if len(signal) < 8:
        return {
            "fft_peak_freq": 0,
            "fft_peak_amp": 0,
            "fft_energy": 0,
            "fft_band_energy_100_1500": 0
        }

    signal = signal - np.mean(signal)

    fft_vals = np.abs(rfft(signal))
    freqs = rfftfreq(len(signal), 1/fs)

    band = (freqs > 100) & (freqs < 1500)
    band_energy = np.sum(fft_vals[band] ** 2)

    peak_idx = np.argmax(fft_vals[1:]) + 1

    return {
        "fft_peak_freq": freqs[peak_idx],
        "fft_peak_amp": fft_vals[peak_idx],
        "fft_energy": np.mean(fft_vals**2),
        "fft_band_energy_100_1500": band_energy
    }