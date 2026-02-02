import csv
from datetime import datetime
from igbt_vibration_stream import stream_acceleration
from igbt_vibration_sliding_window import sliding_window_accel
from igbt_vibration_feature_extraction import extract_vibration_features
from igbt_vibration_health_labelling import assign_vibration_health_label

def real_time_vibration_monitor(sensor, fs=1000, filename="vibration_RF_dataset.csv"):
    stream = stream_acceleration(sensor, sample_period=1/fs)
    windows = sliding_window_accel(stream)

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)

        while True:
            t, ax, ay, az = next(windows)

            if t[-1] - t[0] < 5:
                continue

            features = extract_vibration_features(t, ax, ay, az, fs)
            label = assign_vibration_health_label(
                features["vib_rms"],
                features["vib_kurtosis"]
            )

            row = [
                datetime.now().isoformat(),
                *features.values(),
                label
            ]

            writer.writerow(row)
            f.flush()

            print(f"[{label}] RMS={features['vib_rms']:.4f} m/s²")
