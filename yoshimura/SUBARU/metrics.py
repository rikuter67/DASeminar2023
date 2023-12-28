import numpy as np


def frame_error(pred: float, true: float) -> float:
    error = np.abs(pred - true) / (0.07 * true + 3.)
    if error < 1.0:
        return 1.0
    else:
        return error