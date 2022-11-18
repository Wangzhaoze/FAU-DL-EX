import numpy as np
from .Base import BaseLayer


class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.p = probability
        self.activations = 0
    pass

    def forward(self, input_tensor):
        # only in training phase multiply activations by （1/p)
        if self.testing_phase:
            return input_tensor
        else:
            self.activations = np.random.binomial(1, self.p, input_tensor.shape) / self.p
            return self.activations * input_tensor


    def backward(self, error_tensor):
        return self.activations * error_tensor

