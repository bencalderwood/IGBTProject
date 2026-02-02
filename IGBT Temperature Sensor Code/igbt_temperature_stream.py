import time
import numpy as np

def stream_temperature(sensor, sample_period = 1.0):
    """Generator yielding time-stamped temperature samples"""
    t0 = time.time()
    while True:
        t = time.time() - t0
        T = sensor.temperature
        yield t, T
        time.sleep(sample_period)
