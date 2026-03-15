import numpy as np
import time
import random
from master_feature_extractor import extract_all_features
from csv_writer import write_features_to_csv
#from database_logger import log_features_to_db


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
last_print_time = 0
PRINT_INTERVAL = 0.5

# ========================
# SPI – PT100 Temperature Sensor (MAX31865)
# ========================
#spi = board.SPI()
#cs = digitalio.DigitalInOut(board.D5)

#pt100 = adafruit_max31865.MAX31865(
 #   spi,
  #  cs,
   # wires=3,
    #rtd_nominal=100,
    #ref_resistor=430
#)

# ========================
# I2C – Accelerometer (ADXL345)
# ========================
#i2c = busio.I2C(board.SCL, board.SDA)
#accelerometer = adafruit_adxl34x.ADXL345(i2c)

# Optional: set range for vibration
#accelerometer.range = adafruit_adxl34x.Range.RANGE_16_G

# ========================
# USB CDC – STM32
# ========================
#SERIAL_PORT = "/dev/ttyACM0"   # adjust if needed
#BAUDRATE = 115200

#ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)

#latest_vce = 0.0
#latest_ic = 0.0
#lock = threading.Lock()




# ========================
# SENSOR READ FUNCTIONS
# ========================
#def stm32_reader():
 #   global latest_vce, latest_ic

  #  while True:
   #     try:
    #        line = ser.readline().decode("utf-8").strip()
     #       if not line:
      #          continue

            # Expected format: "12.34, 5.67"
       #     line = line.replace("V=", "").replace(" V", "")
        #    line = line.replace("I=", "").replace(" A", "")
         #   parts = line.split(",")

          #  vce_str = parts[0].strip()
           # ic_str = parts[1].strip()

            #with lock:
             #   latest_vce = float(vce_str)
              #  latest_ic = float(ic_str)

        #except Exception:
         #   continue

# Start reader thread ONCE
#threading.Thread(target=stm32_reader, daemon=True).start()

def read_pt100():
    temp = random.uniform(18.0,35.0)
    return temp


def read_adxl345():
     ax = random.uniform(-2.0,2.0)
     ay = random.uniform(-2.0,2.0)
     az = random.uniform(-2.0,2.0)
     return ax,ay,az

ax, ay, az = read_adxl345()


# Voltage and Current Sensors (ADCs)
def read_vce():
    vce = random.uniform(0.0,15.0)
    return vce



def read_current():
    ic = random.uniform(0.0, 3.0)
    return ic


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
# PROCESS ONE WINDOW AND WRITE FEATURES TO CSV. FILE AND DATABASE
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
    #log_features_to_db(features)

# ========================
# LOG ONE SAMPLE
# ========================
def log_sensors():
     data_buffer["temp"].append(read_pt100())

     ax, ay, az = read_adxl345()
     data_buffer["ax"].append(ax)
     data_buffer["ay"].append(ay)
     data_buffer["az"].append(az)

     data_buffer["vce"].append(read_vce())
     data_buffer["ic"].append(read_current())

# ========================
# MAIN REAL-TIME LOOP
# ========================
while True:

        start = time.perf_counter()

        log_sensors()

        if len(data_buffer["temp"]) >= N_SAMPLES:
            process_window(data_buffer)
            for k in data_buffer:
                data_buffer[k].clear()
        # Print Sensor readings
        current_time = time.perf_counter()

        if current_time - last_print_time >= PRINT_INTERVAL:

            if data_buffer["temp"]:
                print(f"Temperature: {data_buffer['temp'][-1]:.2f} C")

                ax = data_buffer["ax"][-1]
                ay = data_buffer["ay"][-1]
                az = data_buffer["az"][-1]
                print(f"Accelerometer: X= {ax:.2f} g  Y={ay:.2f} g  Z={az:.2f} g")

                print(f"Vce: {data_buffer['vce'][-1]:.2f} V  Ic: {data_buffer['ic'][-1]:.2f} A")
                print("-------------------------")

            last_print_time = current_time

        elapsed = time.perf_counter() - start
        time.sleep(max(0.0, (1.0 / FS) - elapsed))
