
#Stochastic Gradient Descent
class Sgd:
    def __init__(self, learning_rate):
        # learning_rate is a float datatype number
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        #updated weight = weight - gradient * learning_rate
        new_weight = weight_tensor - gradient_tensor * self.learning_rate
        return new_weight
