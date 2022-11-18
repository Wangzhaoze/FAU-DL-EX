
import numpy as np


class Sgd:
    def __init__(self, learning_rate):
        # learning_rate is a float datatype number
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        #updated weight = weight - gradient * learning_rate
        new_weight = weight_tensor - self.learning_rate * gradient_tensor
        return new_weight


class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v_k = 0
        pass

    def calculate_update(self, weight_tensor, gradient_tensor):

        # v_k = momentum * v_k - learning_rate * gradient
        self.v_k = self.momentum_rate * self.v_k - self.learning_rate * gradient_tensor

        weight_updated = weight_tensor + self.v_k

        return weight_updated


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v_k = 0
        self.r_k = 0
        self.k = 1

        pass

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v_k = self.mu * self.v_k + (1 - self.mu) * gradient_tensor
        v_k_hat = self.v_k / (1 - np.power(self.mu, self.k))

        self.r_k = self.rho * self.r_k + (1 - self.rho) * gradient_tensor**2
        r_k_hat = self.r_k / (1 - np.power(self.rho, self.k))

        self.k += 1

        weight_updated = weight_tensor - self.learning_rate * v_k_hat / (np.sqrt(r_k_hat) + np.finfo(float).eps)
        return weight_updated
        pass
