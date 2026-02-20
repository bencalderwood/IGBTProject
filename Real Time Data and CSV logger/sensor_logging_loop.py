import time
import numpy as np
import board
import busio
import digitalio
import adafruit_max31865
import adafruit_adxl34x
from master_feature_extractor import extract_all_features
from csv_writer import write_features_to_csv

#sensor_logging_loop.py starts
#│
#├── reads sensors in real time
#│
#├── fills buffer
#│
#├── buffer full → extract_all_features()
#│
#├── adds timestamp + HealthState
#│
#├── write_features_to_csv(features)
#│     ├── CSV doesn’t exist → init_csv()
#│     └── append row
#│
#└── loop continues forever


# ========================
# CONFIG
# ========================
FS = 200                # Hz
WINDOW_SEC = 1.0
N_SAMPLES = int(FS * WINDOW_SEC)

# ========================
# SPI – PT100 (MAX31865)
# ========================
spi = board.SPI()
cs = digitalio.DigitalInOut(board.D5)

pt100 = adafruit_max31865.MAX31865(
    spi,
    cs,
    wires=3,           # change if 2-wire or 4-wire
    rtd_nominal=100,
    ref_resistor=430
)

# ========================
# I2C – ADXL345
# ========================
i2c = busio.I2C(board.SCL, board.SDA)
accelerometer = adafruit_adxl34x.ADXL345(i2c)

# Optional: set range for vibration
accelerometer.range = adafruit_adxl34x.Range.RANGE_16_G

# ========================
# SENSOR READ FUNCTIONS
# ========================
def read_pt100():
    return pt100.temperature


def read_adxl345():
    """Read accelerometer in g"""
    ax, ay, az = accelerometer.acceleration
    # ADXL345 library returns m/s² → convert to g
    g = 9.80665
    return ax / g, ay / g, az / g

# PLACEHOLDERS (until ADC added)
def read_vce():
    return 0.0


def read_current():
    return 0.0

# ========================
# DATA BUFFER
# ========================
data_buffer = {
    "temp": [],
    "ax": [], "ay": [], "az": [],
    "vce": [],
    "ic": []
}

# ========================
# LOG ONE SAMPLE
# ========================
def log_sensors():
    # Replace with real sensor reads
    temp = read_pt100()
    ax, ay, az = read_adxl345()
    vce = read_vce()
    ic = read_current()

    data_buffer["temp"].append(temp)
    data_buffer["ax"].append(ax)
    data_buffer["ay"].append(ay)
    data_buffer["az"].append(az)
    data_buffer["vce"].append(vce)
    data_buffer["ic"].append(ic)

    # ========================
    # PROCESS ONE WINDOW
    # ========================
    def process_window(buffer):
        data_np = {k: np.array(v) for k, v in buffer.items()}

        # Equal length + NaN safety
        n = min(len(v) for v in data_np.values())
        for key in data_np:
            data_np[key] = np.nan_to_num(data_np[key][:n])

        features = extract_all_features(data_np, fs=FS)
        features["timestamp"] = time.time()
        features["HealthState"] = "Healthy"

        write_features_to_csv(features)

        # ========================
        # MAIN REAL-TIME LOOP
        # ========================
    while True:
        start = time.perf_counter()

        log_sensors()

        if len(data_buffer["temp"]) >= N_SAMPLES:
            process_window(data_buffer)

            # Clear buffer
            for k in data_buffer:
                data_buffer[k].clear()

        elapsed = time.perf_counter() - start
        sleep_time = max(0.0, (1.0 / FS) - elapsed)
        time.sleep(sleep_time)
