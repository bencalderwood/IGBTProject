from collections import deque
import numpy as np

def sliding_window(stream, window_duration=60):
    buffer_t = deque()
    buffer_T = deque()

    for t, T in stream:
        buffer_t.append(t)
        buffer_T.append(T)

        # Remove old samples
        while buffer_t[0] < t - window_duration:
            buffer_t.popleft()
            buffer_T.popleft()

        yield np.array(buffer_t), np.array(buffer_T)
