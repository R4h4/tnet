# TNet - A Python deep learning framework on Numpy

This repository is work in progress. 
One finished it will provide an API to use the most common Neural Network 
architectures. 

Current status:

Layer-Types:
  - Dense Layer / Fully-Connected layer (Status: 80% done)
  - Convolution Layer (Status: 0%)
  - Recurring Layer (Status: 0%)

Activation Functions:
  - Gradient Decent
     - Batch-GD (100%)
     - SGD (50%)
  - Adam Optimizer (0%)
  
## How to build and train your neural network:
It starts with initializing a Neural network. A neural network consaists of a list of layers and a cost function. 
Each layer takes the following parameters:

   `size`: Int, The number of nodes in this layer. For the last layer, this has to be the output size.
   `activation`: Function, The activation function for this layer.
   `input_size`: Int, Required on the first first layer, needs to be equal to the size of the input vector for each set.

After the nn is initialized, we can load the data with `.load(input, target)`
Next, we need to manually initialize the weights and last put not least, we start the training with the parameters:
   `n_epochs`: Int, Number auf trainings iterations
   `alpha`: Float, Learning-Rate

Example:

```
nn = NeuralNetwork([Dense(100, sigmoid, input_size=784), Dense(10, softmax)], cost_function=CrossEntropy())
nn.load_data(x, y)
nn.init_weights()

nn.train(500, alpha=0.3)
```

After the training finished, we can take a look at the error over time using `.plot_error()`.