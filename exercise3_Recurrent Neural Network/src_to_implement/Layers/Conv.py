from .Base import BaseLayer
import numpy as np
import scipy.signal
from copy import deepcopy


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.kernel_shape = convolution_shape
        self.num_kernels = num_kernels
        self.input_tensor = []
        self.bias = np.random.random(num_kernels)

        # 1D convolution or 2D convolution
        self.kc, self.kh, self.kw = [], [], []
        if len(convolution_shape) == 2:
            self.kc, self.kh = convolution_shape
            self.kw = 1
            self.weights = np.random.rand(num_kernels, self.kc, self.kh)
        else:
            self.kc, self.kh, self.kw = convolution_shape
            self.weights = np.random.rand(num_kernels, self.kc, self.kh, self.kw)

        # properties
        self._optimizer_weights = None
        self._optimizer_bias = None
        self._gradient_weights = None
        self._gradient_bias = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        output_tensor = []

        # 2D convolution: (batch_size, channels, height, width)
        if len(input_tensor.shape) == 4:

            # parameters
            batch_size, channel, input_h, input_w = input_tensor.shape
            kc, kh, kw = self.kernel_shape
            sh, sw = self.stride_shape

            # calculate output size
            output_h = int((input_h - 1) / sh + 1)
            output_w = int((input_w - 1) / sw + 1)
            output_tensor = np.zeros((batch_size, self.num_kernels, output_h, output_w))

            # calculate convolution
            for b in range(batch_size):
                for n in range(self.num_kernels):
                    conv = []
                    for c in range(channel):
                        conv.append(scipy.signal.correlate(input_tensor[b, c], self.weights[n, c], mode='same'))

                    output_tensor[b, n, :, :] = np.sum(np.asarray(conv), axis=0)[::sh, ::sw] + self.bias[n]

        # 1D convolution
        if len(input_tensor.shape) == 3:

            # parameters
            batch_size, channel, input_y = input_tensor.shape
            kc, ky = self.kernel_shape
            sy = self.stride_shape[0]

            # calculate output size
            output_y = int((input_y - 1) / sy + 1)
            output_tensor = np.zeros((batch_size, self.num_kernels, output_y))

            # calculate convolution
            for b in range(batch_size):
                for n in range(self.num_kernels):
                    conv = []
                    for c in range(channel):
                        conv.append(scipy.signal.correlate(input_tensor[b, c], self.weights[n, c], mode='same'))
                    output_tensor[b, n, :] = np.sum(np.asarray(conv), axis=0)[::sy] + self.bias[n]

        return output_tensor
        pass

    def backward(self, error_tensor):

        # if 1D: append array to 2D
        conv1d = len(error_tensor.shape) == 3
        if conv1d:
            error_tensor = error_tensor[:, :, :, np.newaxis]
            self.weights = self.weights[:, :, :, np.newaxis]
            self.stride_shape = (*self.stride_shape, 1)
            self.input_tensor = self.input_tensor[:, :, :, np.newaxis]

        # parameters
        batch_size, num_kernels = error_tensor.shape[:2]
        batch, channel, height, width = np.shape(self.input_tensor)
        error_tensor_expand = np.zeros((batch_size, num_kernels, height, width))
        sh, sw = self.stride_shape

        # expand tensor according to stride shape
        for b in range(batch_size):
            for n in range(num_kernels):
                error_tensor_expand[b, n, ::sh, ::sw] = error_tensor[b, n, :, :]

        # calculate convolution
        output_tensor = np.zeros((batch_size, channel, height, width))
        for b in range(batch_size):
            for c in range(channel):
                conv = []
                for n in range(self.num_kernels):
                    conv.append(scipy.signal.convolve(error_tensor_expand[b, n], self.weights[n, c], mode='same'))

                    output_tensor[b, c, :, :] = np.sum(np.asarray(conv), axis=0)

        padding = np.pad(self.input_tensor, ((0, 0), (0, 0), (int(self.kh / 2), int((self.kh - 1) / 2)),
                                             (int(self.kw / 2), int((self.kw - 1) / 2))))

        # calculate gradient with respect to weights
        self._gradient_weights = np.zeros((batch_size, num_kernels, channel, self.kh, self.kw))
        for b in range(batch_size):
            for n in range(num_kernels):
                for c in range(channel):
                    self._gradient_weights[b, n, c] = scipy.signal.correlate(padding[b, c], error_tensor_expand[b, n],
                                                                             mode='valid')
        self._gradient_weights = np.sum(self._gradient_weights, axis=0)

        # calculate gradient with respect to bias
        self._gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))

        # update weights matrix and bias vector
        if self._optimizer_weights is not None:
            self.weights = self._optimizer_weights.calculate_update(self.weights, self.gradient_weights)
        if self._optimizer_bias is not None:
            self.bias = self._optimizer_bias.calculate_update(self.bias, self.gradient_bias)

        # if 1D: back to its original shape, remove the 4th axis
        if conv1d:
            self.weights = np.squeeze(self.weights, axis=3)
            self.stride_shape = self.stride_shape[:-1]
            self.input_tensor = np.squeeze(self.input_tensor, axis=3)
            output_tensor = np.squeeze(output_tensor, axis=3)

        return output_tensor

    # initialize weights and bias
    def initialize(self, weight_initializer, bias_initializer):
        self.weights = weight_initializer.initialize(self.weights.shape, np.prod(self.kernel_shape),
                                                     self.num_kernels * self.kh * self.kw)
        self.bias = bias_initializer.initialize(self.bias.shape, np.prod(self.kernel_shape),
                                                self.num_kernels * self.kh * self.kw)

    @property
    def optimizer(self):
        return self._optimizer_weights

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer_weights = optimizer
        self._optimizer_bias = deepcopy(optimizer)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias
