import numpy as np


def identity(x, deriv=False):
    if deriv:
        return 1
    return x


def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def softmax(x, deriv=False, axis=1):
    shift_x = x - np.max(x)

    try:
        sm = np.exp(shift_x) / np.sum(np.exp(shift_x), axis=axis, keepdims=True)
    except np.AxisError:
        sm = np.exp(shift_x) / np.sum(np.exp(shift_x), axis=None, keepdims=True)

    if deriv:
        return x * (
                    1 - x)  # https://datascience.stackexchange.com/questions/29735/how-to-apply-the-gradient-of-softmax-in-backprop
    else:
        return sm
