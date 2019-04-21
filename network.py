import numpy as np


class NeuralNetwork:
    # Class Attributes
    train_error = list()

    def __init__(self, layers, cost_function):
        assert isinstance(layers, list), "Input needs to be a list of Layers"
        assert len(layers) > 1, "Input needs to be a list of at least two Layers"
        self.layers = layers
        self.x = np.zeros(1)
        self.target = np.zeros(1)
        self.current_state = np.zeros(1)
        assert callable(error_function), "Chose a valid error function"
        self.error_function = error_function
        self.l_error = list()  # Error over time is saved here
        self.cost = cost_function

    def load_data(self, x: np.ndarray, target: np.ndarray):
        # Check if input and output have the same amount of cases
        assert len(x) == len(target), f"Input and target output contain a different number of cases " \
            f"({len(x)} vs. {len(target)})"
        # Check if x and target are numeric numpy arrays
        assert np.issubdtype(x.dtype, np.number) & np.issubdtype(target.dtype, np.number), \
            "Both input and target need to be numeric arrays"

        self.x = x
        self.target = target

    def init_weights(self):
        # First we infer the input size for each of the layer, except the first one
        for i, layer in enumerate(self.layers):
            if i == 0:
                assert layer.input_size, "The first layer need to be initialized with the parameter 'input_size'"
            else:
                layer.input_size = self.layers[i - 1].size

        # Initialize the weights with random noise
        np.random.RandomState(42)
        sigma = 0.03

        # Then we initialize the weights by using the input size (+1 bias Unit) and amount of units
        for layer in self.layers:
            layer.weights = sigma * np.random.randn(layer.input_size + 1, layer.size)

    def train(self, n_epochs: int, alpha=0.01):
        for epoch in tqdm(range(n_epochs)):
            # Calculate forward
            self.current_state = calc_output(self.x)

            mse_epoch = mse(self.current_state, self.target)
            self.l_error.append(mse_epoch)

            # Calculate backwards
            # Start with calculating the error/loss at each layer
            for i, layer in enumerate(reversed(self.layers)):
                if i == 0:
                    layer.error = np.subtract(self.current_state, self.target)
                    # Start with calculating the error/loss at the output
                else:
                    layer.calc_error(prev_error=self.layers[len(self.layers) - i].error,
                                     prev_weights=self.layers[len(self.layers) - i].weights)

            # Then calculate the partial derivative and update the weights
            for layer in self.layers:
                layer.update_weights(alpha)

    def train_sgd(self, n_epochs: int, alpha=0.01):
        for epoch in tqdm(range(n_epochs)):
            # Shuffle the examples in the beginning of each epoch:
            self.x, self.target = unison_shuffled_copies(self.x, self.target)
            for i_step, example in enumerate(self.x):
                # Calculate forward
                self.current_state = example
                for layer in self.layers:
                    layer.forward(self.current_state)
                    self.current_state = layer.activations_out

                mse_epoch = mse(self.current_state, self.target[i_step])
                self.l_error.append(mse_epoch)
                # if not (epoch % 10):
                #    if mse_epoch > (min(self.l_error) * 1.1):
                #        alpha = alpha/2
                #        print("Devide alpha by 2")

                # Calculate backwards
                # Start with calculating the error/loss at each layer
                for i, layer in enumerate(reversed(self.layers)):
                    if i == 0:
                        layer.error = np.subtract(self.current_state, self.target[i_step])
                        # Start with calculating the error/loss at the output
                    else:
                        layer.calc_error(prev_error=self.layers[len(self.layers) - i].error,
                                         prev_weights=self.layers[len(self.layers) - i].weights)

                # Then calculate the partial derivative and update the weights
                for layer in self.layers:
                    layer.update_weights(alpha)

    def plot_error(self):
        plt.plot(range(len(self.l_error)), self.l_error)
        plt.show()

    def calc_output(self, _input):
        # Calculate
        current_state = _input
        for layer in self.layers:
            layer.forward(self.current_state)
            current_state = layer.activations_out
        return current_state


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
