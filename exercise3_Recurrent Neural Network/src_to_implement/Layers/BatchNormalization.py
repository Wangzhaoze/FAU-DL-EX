from .Base import BaseLayer
import numpy as np
from copy import deepcopy
from .Helpers import compute_bn_gradients


class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self.c = channels
        self.input_tensor = np.array([])
        self.shape = tuple()   # shape of input_tensor
        self.image2vec = False

        # define initialize weights(gamma) and bias(beta)
        self.weights = np.ones(self.c)
        self.bias = np.zeros(self.c)

        # define optimizer properties
        self._optimizer_weights = None
        self._optimizer_bias = None

        #define mean and variance
        self.mean = 0
        self.variance = 0
        self.X_wave = 0

        # define gradients
        self.gradient_input = None
        self.gradient_weights = None
        self.gradient_bias = None

    def reformat(self, input_tensor):
        """
        reshape input_tensor between 4-D with 2-D
        :param input_tensor: input vector(2-D) or matrix(4-D)
        :return: reformed result
        """
        output = []
        # image-like input tensor
        if len(input_tensor.shape) == 4:
            self.shape = input_tensor.shape
            b, c, h, w = input_tensor.shape
            output = input_tensor.swapaxes(0, 1).reshape((c, -1)).T

        # vector-like input tensor
        elif len(input_tensor.shape) == 2:
            b, c, h, w = self.shape
            output = input_tensor.T.reshape(c, b, h, w).swapaxes(0, 1)

        return output


    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        alpha = 0.8

        if len(input_tensor.shape) == 4:
            self.image2vec = True
            input_tensor = self.reformat(input_tensor)

        # too expensive to calculate mean and var in test, so we use moving average
        if self.testing_phase:
            self.mean = alpha * self.mean + (1 - alpha) * self.mean
            self.variance = alpha * self.variance + (1 - alpha) * self.variance
        else:
            self.mean = np.mean(input_tensor, axis=0)
            self.variance = np.var(input_tensor, axis=0)

        self.X_wave = (input_tensor - self.mean) / np.sqrt(self.variance + np.finfo(float).eps)
        output = self.X_wave * self.weights + self.bias

        if self.image2vec:
            output = self.reformat(output)
        return output


    def backward(self, error_tensor):
        if self.image2vec:
            error_tensor = self.reformat(error_tensor)
            self.input_tensor = self.reformat(self.input_tensor)

        # calculate gradients w.r.t input, weights and bias
        self.gradient_input = compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.mean, self.variance)
        self.gradient_weights = np.sum(error_tensor * self.X_wave, axis=0)
        self.gradient_bias = np.sum(error_tensor, axis=0)

        # update weights and bias
        if self._optimizer_weights is not None:
            self.weights = self._optimizer_weights.calculate_update(self.weights, self.gradient_weights)
        if self._optimizer_bias is not None:
            self.bias = self._optimizer_bias.calculate_update(self.bias, self.gradient_bias)

        if self.image2vec:
            self.gradient_input = self.reformat(self.gradient_input)

        return self.gradient_input

    def initialize(self, weight_initializer, bias_initializer):
        self.weights = weight_initializer.initialize(self.weights.shape, self.c, self.c)
        self.bias = bias_initializer.initialize(self.bias.shape, self.c, self.c)
        pass

    @property
    def optimizer(self):
        return self._optimizer_weights

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer_weights = optimizer
        self._optimizer_bias = deepcopy(optimizer)

