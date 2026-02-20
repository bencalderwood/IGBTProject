import numpy as np
from temperature_feature_extraction import extract_temperature_features
from accelerometer_feature_extraction import magnitude
from accelerometer_feature_extraction import extract_vibration_features
from accelerometer_feature_extraction import extract_fft_features
from voltage_feature_extraction import extract_vce_features
from current_feature_extraction import extract_current_features
from power_feature_extraction import extract_power_features


def extract_all_features(data, fs):

    # Ensures equal length and remove NaNs from sensor data
    n = min(len(v) for v in data.values())
    for k in data:
        data[k] = np.nan_to_num(data[k][:n])

    features = {}

    mag = magnitude(data["ax"], data["ay"], data["az"]) #magnitude features

    features.update(extract_temperature_features(data["temp"], fs)) #temperature features
    features.update(extract_vibration_features(data["ax"], data["ay"], data["az"])) # accel/vib features
    features.update(extract_fft_features(mag, fs)) # vibration FFT features
    features.update(extract_vce_features(data["vce"], fs)) #voltage features
    features.update(extract_current_features(data["ic"], fs)) # current features
    features.update(extract_power_features(data["vce"], data["ic"])) #power features(from vce and ic)

    return features