
#Base Layer should be inherited by every layer in framework
class BaseLayer:
    def __init__(self):
        self.trainable = False

    def forward(self, input_tensor):
        pass

    def backward(self, error_tensor):
        pass
