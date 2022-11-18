from src_to_implement.Layers.Base import BaseLayer
import numpy as np


class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_shape = 0
        pass

    def forward(self, input_tensor):
        '''
        reshape input_tensor into a vector
        :param input_tensor: matrix
        :return: reshaped vector
        '''

        # parameters
        self.input_shape = input_tensor.shape
        batch_size = input_tensor.shape[0]
        vector_shape = input_tensor.shape[1:]

        # reshape matrix into batch_size vectors
        output_tensor = input_tensor.reshape(batch_size, np.prod(vector_shape))

        return output_tensor


    def backward(self, error_tensor):
        '''
        reshape vector back into its original shape
        :param error_tensor: backward error_tensor from next layer, vector
        :return: reshaped matrix
        '''

        backward_tensor = error_tensor.reshape(self.input_shape)
        return backward_tensor
        pass

