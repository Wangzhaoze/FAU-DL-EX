import numpy as np
from src_to_implement.Layers.Base import BaseLayer



class FullyConnected(BaseLayer):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_tensor = 0

        #define a protected member 'optimizer'
        self._optimizer = 0

        #initialize the matrix 'weights' : the combination of matrix weight and vector bias
        self.weight = np.random.uniform(low=0.0, high=1.0, size=(input_size, output_size))
        self.bias = np.random.uniform(low=0.0, high=1.0, size=(1, output_size))

        self.weights = np.row_stack((self.weight, self.bias))

        #returns the gradient with respect to the weights, after they have been calculated in the backward().
        self.gradient_weights = 0

        self.input_size = input_size
        self.output_size = output_size

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, set_value):
        self._optimizer = set_value


    def initialize(self, weights_initializer, bias_initializer):
        '''
        initialize weights and bias
        :param weights_initializer: weights class with method 'initialize'
        :param bias_initializer: bias class with method 'initialize'
        :return: None
        '''

        self.bias = bias_initializer.initialize((1, self.output_size), 1, self.output_size)
        self.weight = weights_initializer.initialize((self.input_size, self.output_size),
                                                     self.input_size,
                                                     self.output_size)
        self.weights = np.row_stack((self.weight, self.bias))
        pass

    def forward(self, input_tensor):
        """

        :param input_tensor: a matrix with input size columns and batch size rows
        :return: a tensor that serves as the input tensor for the next layer
        """
        #define a column vector of ones
        column_ones = np.ones((input_tensor.shape[0], 1))

        #X is a matrix with rightest column are ones
        self.input_tensor = np.column_stack((input_tensor, column_ones))

        #according to the Lecture should be Y = Weights * X
        #but here we should set Y = X * Weights, otherwise the size of output matrix will be wrong
        output_tensor = np.dot(self.input_tensor, self.weights)
        return output_tensor

    def backward(self, error_tensor):
        """

        :param error_tensor: backward output from the next layer.
        :return: error_tensor for the previous layer.
        """
        #gradient with respect to the weights
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)

        #gradient with respect to the input
        output_error = np.dot(error_tensor, self.weight.T)

        #update the weights matrix
        if self._optimizer != 0:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
        return output_error

