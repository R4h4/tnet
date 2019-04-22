import numpy as np


class CostFunction:
    # Define the error functions
    # Each error function can be called directly or for its error
    def __call__(self, predictions, targets):
        raise NotImplementedError

    def error(self, predictions, target):
        raise NotImplementedError


class SquaredError(CostFunction):

    def __call__(self, predictions, targets, ax=None):
        return (np.square(predictions - targets) / 2).mean(axis=ax)

    def error(self, predictions, targets):
        return targets - predictions


class CrossEntropy(CostFunction):

    def __init__(self):
        self.epsilon = 1e-12

    def __call__(self, predictions, targets, ax=None):
        # The epsilon prevents log(targets) to be inf
        clipped = np.clip(predictions, self.epsilon, 1. - self.epsilon)
        N = clipped.shape[0]
        return -np.sum(targets * np.log(clipped + 1e-9)) / N

    def error(self, predictions, targets):
        # https://gist.github.com/Atlas7/22372a4f6b0846cfc3797766d7b529e8
        if predictions.ndim == 1:
            n_samples = 1
        else:
            n_samples = len(predictions)
        return -(1.0 / n_samples) * np.sum(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
