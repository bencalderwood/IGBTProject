from scipy.stats import kurtosis
from numpy.fft import rfft, rfftfreq
import numpy as np

def extract_vibration_features(t, ax, ay, az, fs):
    # Resultant acceleration
    a = np.sqrt(ax**2 + ay**2 + az**2)

    # Time-domain features
    rms = np.sqrt(np.mean(a**2))
    peak = np.max(np.abs(a))
    crest_factor = peak / rms
    std = np.std(a)
    kurt = kurtosis(a)

    # Frequency-domain features
    freqs = rfftfreq(len(a), d=1/fs)
    spectrum = np.abs(rfft(a))

    band_energy = np.sum(spectrum[(freqs > 10) & (freqs < 500)])
    dominant_freq = freqs[np.argmax(spectrum)]

    return {
        "vib_resultant acceleration": a,
        "vib_rms": rms,
        "vib_peak": peak,
        "vib_crest": crest_factor,
        "vib_std": std,
        "vib_kurtosis": kurt,
        "vib_band_energy": band_energy,
        "vib_dom_freq": dominant_freq,
    }
