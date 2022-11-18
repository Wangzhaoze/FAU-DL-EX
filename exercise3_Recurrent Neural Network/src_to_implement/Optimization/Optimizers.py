
import numpy as np


class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer


class Sgd(Optimizer):
    def __init__(self, learning_rate):
        super().__init__()
        # learning_rate is a float datatype number
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        #updated weight = weight - gradient * learning_rate
        weight_updated = weight_tensor - self.learning_rate * gradient_tensor

        #updated_weight = weight - earning_rate * Loss w.r.t weight
        if self.regularizer is not None:
            weight_updated -= self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        return weight_updated



class SgdWithMomentum(Optimizer):
    #增加矩，提高下降速度，有一定概率冲过局部最小值
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v_k = 0
        pass

    def calculate_update(self, weight_tensor, gradient_tensor):

        # v_k = momentum * v_k - learning_rate * gradient
        self.v_k = self.momentum_rate * self.v_k - self.learning_rate * gradient_tensor
        weight_updated = weight_tensor + self.v_k

        # updated_weight = weight - earning_rate * Loss w.r.t weight
        if self.regularizer is not None:
            weight_updated -= self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        return weight_updated


class Adam(Optimizer):
    #在增加矩的基础上增加一个参数控制学习率，更新时考虑之前的权重值
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
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

        # updated_weight = weight - earning_rate * Loss w.r.t weight
        if self.regularizer is not None:
            weight_updated -= self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)

        return weight_updated
        pass
