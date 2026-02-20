import csv
import os


CSV_FILE = "igbt_features.csv"
HEADER_WRITTEN = False


def init_csv(feature_keys):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=feature_keys)
        writer.writeheader()


def write_features_to_csv(features):
    global HEADER_WRITTEN

    if not os.path.exists(CSV_FILE) or not HEADER_WRITTEN:
        init_csv(features.keys())
        HEADER_WRITTEN = True

    # Ensure all keys exist (NaN-safe)
    row = {k: features.get(k, 0) for k in features.keys()}

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writerow(row)