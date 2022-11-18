from copy import deepcopy
import numpy as np


class NeuralNetwork:
    def __init__(self, optimizer):

        # received upon construction as the first argument
        self.optimizer = optimizer
        # contain the loss value for each iteration after calling train
        self.loss = list()
        # hold the architecture
        self.layers = list()
        # provide input data and labels
        self.data_layer = None
        # referring to the special layer providing loss and prediction
        self.loss_layer = None
        self._labels = None

    def forward(self):
        """
        using input from the data layer and passing it through all layers of the network.
        :return: output of the last layer (i. e. the loss layer) of the network.
        """
        #get input_tensor and labels bu using next()
        input_tensor, self._labels = self.data_layer.next()

        #passing through layers in self.layers
        for each_layer in self.layers:
            input_tensor = each_layer.forward(input_tensor)

        #passing through the last layer: loss_layer
        input_tensor = self.loss_layer.forward(input_tensor, self._labels)
        return input_tensor

    def backward(self):
        """
        starting from the loss_layer and propagating it back through the network.
        :return: error_tensor after backpropagation through the network
        """
        error_tensor = self.loss_layer.backward(self._labels)
        for each_layer in self.layers[::-1]:
            error_tensor = each_layer.backward(error_tensor)

        return error_tensor

    def append_layer(self, layer):
        '''
        If the layer is trainable, make a deepcopy and set it for the layer
        :param layer: a trainable or non-trainable layer
        :return: list of layers which are trainable and after deepcopy of the neural network’s optimizer
        '''
        if layer.trainable:
            layer.optimizer = deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        """
        trains the network for iterations and stores the loss for each iteration.
        :param iterations: times of iterations
        :return:
        """
        for i in range(iterations):
            self.loss.append(np.sum(np.square(self.forward())))
            self.backward()

    def test(self, input_tensor):
        """
        :param input_tensor: input_tensor which is propagated through the network
        :return: the prediction of the last layer
        """
        for each_layer in self.layers:
            input_tensor = each_layer.forward(input_tensor)
        return input_tensor
