import csv
from datetime import datetime
from igbt_temperature_stream import stream_temperature
from igbt_sliding_window import sliding_window
from igbt_temperature_feature_extraction import extract_temperature_features
from igbt_temperature_health_labelling import assign_temperature_health_label

def real_time_temperature_monitor(sensor, test_config, filename="temperature_RF_dataset.csv"):
    stream = stream_temperature(sensor)
    windows = sliding_window(stream)

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)

        while True:
            t, T = next(windows)

            # Only extract when window is full
            if t[-1] - t[0] < 60:
                continue

            features = extract_temperature_features(t, T)
            label = assign_temperature_health_label(features["T_mean"])

            row = [
                # ---- Manual test parameters ----
                test_config["switching_frequency_khz"],
                test_config["gate_voltage_V"],
                test_config["pwm_duty_cycle"],
                test_config["load_current_A"],

                datetime.now().isoformat(),
                features["T_mean"],
                features["T_max"],
                features["T_std"],
                features["mean_dTdt"],
                features["max_dTdt"],
                features["DeltaT_steady_state"],
                features["time_below_or_equal_75"],
                features["time_at_75_to_125"],
                features["time_above_125"],
                label
            ]

            writer.writerow(row)
            f.flush()

            print(
                f"[{label}] "
                f"T_mean={features['T_mean']:.2f} °C | "
                f"Vg={test_config['gate_voltage_v']} V | "
                f"f_sw={test_config['switching_frequency_khz']} kHz"
            )
