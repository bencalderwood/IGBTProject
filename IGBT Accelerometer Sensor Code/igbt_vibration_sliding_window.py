from collections import deque
import numpy as np

def sliding_window_accel(stream, window_duration=5):
    buffer_t = deque()
    buffer_ax = deque()
    buffer_ay = deque()
    buffer_az = deque()

    for t, ax, ay, az in stream:
        buffer_t.append(t)
        buffer_ax.append(ax)
        buffer_ay.append(ay)
        buffer_az.append(az)

        while buffer_t[0] < t - window_duration:
            buffer_t.popleft()
            buffer_ax.popleft()
            buffer_ay.popleft()
            buffer_az.popleft()

        yield (
            np.array(buffer_t),
            np.array(buffer_ax),
            np.array(buffer_ay),
            np.array(buffer_az),
        )
