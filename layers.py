import numpy as np

class Layer:

    def __init__(self, size: int, activation, input_size=False):
        assert isinstance(size, int), "The number of nodes needs to be of type int"
        self.size = size
        assert callable(activation), "Chose a valid activation function"
        self.activation = activation
        self.activations_in = np.zeros(1)
        self.activations_out = np.zeros(size)
        self.error = np.zeros(size)
        self.weights = np.zeros(size)
        self.isfirst = False
        self.input_size = input_size
        if input_size:
            self.isfirst = True

    def __len__(self):
        return self.size


class Dense(Layer):

    def forward(self, activations_in):
        # Save incoming activations for later backpropagation and add bias unit
        if activations_in.ndim == 1:
            ones_shape = 1
        else:
            ones_shape = (len(activations_in), 1) + activations_in.shape[2:]

        self.activations_in = np.hstack((np.ones(shape=ones_shape), activations_in))
        self.activations_out = self.activation(np.dot(self.activations_in, self.weights))

    def calc_error(self, prev_error, prev_weights):
        self.error = np.dot(prev_error, prev_weights.T[:, 1:]) * self.activation(self.activations_out, deriv=True)

    def update_weights(self, alpha):
        # The first layer does have one weight less due to the missing bias unit
        # Calculate the partial derivatives for the Error in respect to each weight
        if self.isfirst:
            if self.activations_in.ndim == 1:
                partial_derivative = self.activations_in[:, np.newaxis] * self.error[np.newaxis, :]
                gradient = partial_derivative
            else:
                partial_derivative = self.activations_in[:, :1, np.newaxis] * self.error[:, np.newaxis, :]
                gradient = np.average(partial_derivative, axis=0)
        else:
            if self.activations_in.ndim == 1:
                partial_derivative = self.activations_in[:, np.newaxis] * self.error[np.newaxis, :]
                gradient = partial_derivative
            else:
                partial_derivative = self.activations_in[:, :1, np.newaxis] * self.error[:, np.newaxis, :]
                gradient = np.average(partial_derivative, axis=0)
        self.weights += -alpha * gradient


def batch_gd(weights, alpha, gradient):
    return -alpha * gradient + weights


def stochastic_gd():
    pass


def mini_batch_gd():
    pass


class Activision(Layer):

    def forward(self, activations_in):
        self.activations_out = self.activation(np.dot(activations_in, self.weights))

    def backward(self):
        pass
