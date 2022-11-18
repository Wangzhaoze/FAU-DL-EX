import numpy as np


class Constant:
    def __init__(self, constant=0.1):
        self.constant = constant

    def initialize(self, weights_shape, fan_in, fan_out):
        return self.constant * np.ones(weights_shape)


class UniformRandom:
    def __init__(self):
        self.tensor = []

    def initialize(self, weights_shape, fan_in, fan_out):
        self.tensor = np.random.random(weights_shape)
        return self.tensor


class Xavier:
    def __init__(self):
        self.tensor = []

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2) / np.sqrt(fan_in + fan_out)
        self.tensor = np.random.normal(0, sigma, weights_shape)
        return self.tensor


class He:
    def __init__(self):
        self.tensor = []

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2) / np.sqrt(fan_in)
        self.tensor = np.random.normal(0, sigma, weights_shape)
        return self.tensor
