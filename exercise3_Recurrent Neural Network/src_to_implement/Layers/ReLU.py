import numpy as np
from src_to_implement.Layers.Base import BaseLayer

''' Rectified Linear Unit'''


class ReLU(BaseLayer):

    def __init__(self):
        """
        Applies the rectified linear unit function element-wise: relu(x) = max(x, 0)
        """
        super().__init__()
        self.input_tensor = 0

    def forward(self, input_tensor):
        """
        forward method: relu(x) = max(x, 0)
        :param input_tensor: forward output from the previous layer
        :return: input_tensor for the next layer.
        """
        self.input_tensor = input_tensor
        # Record the Input for backward calculation
        return np.maximum(input_tensor, np.zeros_like(input_tensor))
        pass

    def backward(self, error_tensor):
        """
        backward method: e(n-1) = 0           --input_tensor <= 0
                                = e(n-1)      --otherwise
        :param error_tensor: backward output from the next layer
        :return: error tensor for the previous layer.
        """

        error_tensor[np.where(self.input_tensor <= 0)] = 0

        return error_tensor
