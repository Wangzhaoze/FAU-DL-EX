from src_to_implement.Layers.Base import BaseLayer
import numpy as np


class Pooling(BaseLayer):

    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.trainable = False
        self.pool_h = pooling_shape[0]
        self.pool_w = pooling_shape[1]
        self.stride_h = stride_shape[0]
        self.stride_w = stride_shape[1]
        self.input_tensor = []
        self.index = []

        pass

    def forward(self, input_tensor):
        '''
        similar to convolution, select the max value as the pooling result of every block
        :param input_tensor: input matrix
        :return: max-pooling result
        '''

        # parameters
        self.input_tensor = input_tensor
        batch_size, channel, height, width = input_tensor.shape
        output_height = int((height - self.pool_h) / self.stride_h + 1)
        output_width = int((width - self.pool_w) / self.stride_w + 1)
        output = np.zeros((batch_size, channel, output_height, output_width))
        self.index = np.zeros((batch_size, channel, output_height, output_width), dtype=int)

        # max-pooling and save the index, which will be used into backward()
        for b in range(batch_size):
            for c in range(channel):
                for i in range(output_height):
                    for j in range(output_width):
                        output[b, c, i, j] = np.max(
                            input_tensor[b, c, (self.stride_h * i):(self.stride_h * i + self.pool_h),
                            (self.stride_w * j):(self.stride_w * j + self.pool_w)])

                        self.index[b, c, i, j] = np.argmax(
                            input_tensor[b, c, (self.stride_h * i):(self.stride_h * i + self.pool_h),
                            (self.stride_w * j):(self.stride_w * j + self.pool_w)])

        return output
        pass

    def backward(self, error_tensor):
        '''
        restore the matrix according to max-pooling rule
        :param error_tensor: matrix after pooling
        :return: restored matrix
        '''
        # parameters
        batch_size, channel, height, width = error_tensor.shape
        output_tensor = np.zeros_like(self.input_tensor)

        # restore matrix with index of max-value, which are stored in forward()
        for b in range(batch_size):
            for c in range(channel):
                for i in range(height):
                    for j in range(width):
                        block = np.zeros((self.pool_h * self.pool_w, 1))
                        index = self.index[b, c, i, j]
                        block[index, 0] = error_tensor[b, c, i, j]
                        output_tensor[b, c, (self.stride_h * i):(self.stride_h * i + self.pool_h),
                          (self.stride_w * j):(self.stride_w * j + self.pool_w)] += block.reshape((self.pool_h, self.pool_w))

        return output_tensor
        pass


