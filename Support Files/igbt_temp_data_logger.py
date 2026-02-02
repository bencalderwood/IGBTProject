import time
## data logging function for the physical temperature sensor at 2 Hz (should return time and sensor Temp)
def collect_temperature_window(sensor, duration=60, fs=2):
    t_vals = []
    T_vals = []

    start = time.time()

    while time.time() - start < duration:
        t_vals.append(time.time() - start)
        T_vals.append(sensor.temperature)
        time.sleep(1/fs)

    return t_vals, T_vals
