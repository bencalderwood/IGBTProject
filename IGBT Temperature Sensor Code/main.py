from igbt_temperature_stream import stream_temperature
from igbt_sliding_window import sliding_window
from igbt_temperature_feature_extraction import extract_temperature_features
from igbt_temperature_health_labelling import assign_temperature_health_label
from igbt_real_time_csv_logger import real_time_temperature_monitor
from pt100_sensor import init_temperature_sensor

import board
import digitalio
import adafruit_max31865

# ---------- Temperature Sensor setup (may need to edit to your assigned variables) ----------
init_temperature_sensor(
 spi = board.SPI()
 cs = digitalio.DigitalInOut(board.D5)

 sensor = adafruit_max31865.MAX31865(
     spi, cs,
     rtd_nominal=100.0,
     ref_resistor=430.0,
     wires=3
 ))
#
# ---------- Main loop ----------
t, T = stream_temperature(sensor, sample_period = 1.0)#sensor will be correct when you define it with your variables, you may need to create a python file to set up

features = extract_temperature_features(t, T)

health_label = assign_temperature_health_label(features["T_mean"])

test_config = {
    "switching_frequency_khz": 10,   # 10 kHz
    "gate_voltage_v": 15.0,            # V
    "pwm_duty_cycle": 0.6,             # 60%
    "load_current_a": 3            # A
}

real_time_temperature_monitor(sensor, test_config)

real_time_temperature_monitor(
    features,
    health_label,
    filename="temperature_RF_dataset.csv"
)

print("Temperature feature window saved.")