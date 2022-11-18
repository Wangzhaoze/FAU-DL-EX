import numpy as np
from src_to_implement.Layers.Base import BaseLayer


class SoftMax(BaseLayer):
    """
    Normalized the probability distribution of output so that its sum is 1
    """

    def __init__(self):
        super().__init__()
        self.input_tensor = 0

    def forward(self, input_tensor):
        """

        :param input_tensor: forward output from the previous layer
        :return: the estimated class probabilities for each row representing an element of the batch.
        """
        #calculate max value for each row vector of input_tensor
        #reschape as a column used for subtract
        max_column = np.max(input_tensor, axis=1).reshape(input_tensor.shape[0], 1)

        #exponent value of input_tensor after shifting
        exp_x = np.exp(input_tensor - max_column)

        #transformed probabilities can be computed with formula
        tran_probabilities = exp_x / np.sum(exp_x, axis=1).reshape(input_tensor.shape[0], 1)

        self.input_tensor = tran_probabilities

        return tran_probabilities

    def backward(self, error_tensor):
        """
        method: error_tensor(n-1) = y_hat * (error_tensor(n) - sum(error_tensor(n, i) * y_hat(i)))
        :param error_tensor: backward output from the next layer
        :return: error tensor for the previous layer.
        """
        #calculate value of addition part
        #reschape as a column used for subtract
        sum_part = np.sum(error_tensor * self.input_tensor, axis=1).reshape(error_tensor.shape[0], 1)

        output_error = self.input_tensor * (error_tensor - sum_part)
        return output_error

