import time

def stream_acceleration(sensor, sample_period=0.01):
    """
    Generator yielding timestamped acceleration samples (x, y, z)
    """
    t0 = time.time()
    while True:
        t = time.time() - t0
        ax, ay, az = sensor.acceleration  # m/s^2
        yield t, ax, ay, az
        time.sleep(sample_period)
