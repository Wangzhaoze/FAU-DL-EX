from copy import deepcopy

import numpy as np

from .FullyConnected import FullyConnected
from .Sigmoid import Sigmoid
from .TanH import TanH
from .Base import BaseLayer


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        # parameters
        self.trainable = True
        self.input_tensor = []
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # activation function and fully connected layer
        self.TanH = TanH()
        self.Sigmoid = Sigmoid()
        self.hx2u = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.h2o = FullyConnected(self.hidden_size, self.output_size)

        # calculation results in RNN
        self.hidden_state = np.zeros(hidden_size)
        self.h_archive = None
        self.o_archive = None
        self.u_archive = None
        self._memorize = False

        # optimizer and gradients
        self._optimizer_weights_hx2u = None
        self._optimizer_weights_h2o = None
        self._gradient_weights = None
        self.gradient_weights_h2o = None

    def initialize(self, weight_initializer, bias_initializer):
        self.hx2u.initialize(weight_initializer, bias_initializer)
        self.h2o.initialize(weight_initializer, bias_initializer)
        pass

    def forward(self, input_tensor):
        # save the input tensor
        self.input_tensor = input_tensor

        # calculate shape of output matrix
        batch_size = input_tensor.shape[0]
        output = np.zeros((batch_size, self.output_size))

        # define h,o,u matrix, which are used in the calculation later
        self.h_archive = np.zeros((batch_size + 1, self.hidden_size))
        self.o_archive = np.zeros((batch_size, self.output_size))
        self.u_archive = np.zeros((batch_size, self.hidden_size))

        for t in range(batch_size):
            x = input_tensor[t, :]

            # hx2u_input is a combination vector of x and h
            hx2u_input = np.append(x, self.hidden_state).reshape((1, -1))

            # calculate u, h, o according to the formal
            u = self.hx2u.forward(hx2u_input)
            h = self.TanH.forward(u)
            o = self.h2o.forward(h)
            output[t] = self.Sigmoid.forward(o[0])

            # save computation results
            self.hidden_state = list(h.T)
            self.h_archive[t + 1] = list(h.T)
            self.u_archive[t] = u
            self.o_archive[t] = list(o.T)

        # decide whether to ignore what happened in previous iterations ("memorize=False") or not ("memorize=True").
        if not self._memorize:
            self.hidden_state = np.zeros(self.hidden_size)

        return output

    def backward(self, error_tensor):
        # exact time and create reverse time sequence
        time = error_tensor.shape[0]
        reverse_time_sequence = reversed(range(time))

        # define output
        x = np.zeros((time, self.input_size))

        # property gradient_weights in hidden layer
        self._gradient_weights = np.zeros_like(self.weights)

        # gradient_weights in h2o(FCL)
        self.gradient_weights_h2o = np.zeros_like(self.h2o.weights)

        # gradient w.r.t h in hidden layer
        gradient_h = np.zeros((time + 1, self.hidden_size))

        for t in reverse_time_sequence:
            # calculate Sigmoid backwards part
            self.Sigmoid.forward(self.o_archive[t])
            diff_o = self.Sigmoid.backward((error_tensor[t].reshape((1, -1))))

            # calculate h2o backwards part
            self.h2o.forward(self.h_archive[t + 1].reshape((1, -1)))
            o2h_tensor = self.h2o.backward(diff_o)
            self.gradient_weights_h2o += self.h2o.gradient_weights

            # calculate TanH backwards part
            self.TanH.forward(self.u_archive[t])
            diff_u = self.TanH.backward(gradient_h[t + 1] + o2h_tensor)

            # calculate hx2u backwards part
            self.hx2u.forward(np.append(self.input_tensor[t], self.h_archive[t]).reshape((1, -1)))
            u2hx_tensor = self.hx2u.backward(diff_u)[0]
            self._gradient_weights += self.hx2u.gradient_weights

            # x and gradient_h are subset of backwards tensor of u2hx
            x[t] = u2hx_tensor[0:self.input_size]
            gradient_h[t] = u2hx_tensor[self.input_size::]

        # update weights
        if self._optimizer_weights_hx2u is not None:
            self.hx2u.weights = self._optimizer_weights_hx2u.calculate_update(self.hx2u.weights, self._gradient_weights)
        if self._optimizer_weights_h2o is not None:
            self.h2o.weights = self._optimizer_weights_h2o.calculate_update(self.h2o.weights, self.gradient_weights_h2o)

        return x

    @property
    def weights(self):
        return self.hx2u.weights

    @weights.setter
    def weights(self, weights):
        self.hx2u.weights = weights

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights

    @property
    def optimizer(self):
        return self._optimizer_weights_hx2u

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer_weights_hx2u = optimizer
        self._optimizer_weights_h2o = deepcopy(optimizer)

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, memorize):
        self._memorize = memorize
