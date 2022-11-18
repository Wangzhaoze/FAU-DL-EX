import pickle

import numpy as np
from copy import deepcopy


class NeuralNetwork:
    def __init__(self, optimizer, weight_initializer, bias_initializer):
        self.optimizer = optimizer
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        self.loss = list()
        self.layers = list()
        self._label = list()
        self.data_layer = None
        self.loss_layer = None

        self._phase = 0

    def __getstate__(self):
        # must return a dict
        self.data_layer = None
        return self.__dict__

    def __setstate__(self, state):
        self.data_layer = None
        return

    def forward(self):
        """
        using input from the data layer and passing it through all layers of the network.
        loss: the loss of regularization constrain part
        :return: output of the last layer (i. e. the loss layer) of the network.
        """
        # get input_tensor and labels bu using next()
        input_tensor, self._label = self.data_layer.next()

        loss = 0
        # passing through layers in self.layers
        for each_layer in self.layers:
            input_tensor = each_layer.forward(input_tensor)
            if each_layer.trainable and each_layer.optimizer.regularizer is not None:
                loss += each_layer.optimizer.regularizer.norm(each_layer.weights)

        # passing through the last layer: loss_layer
        input_tensor = self.loss_layer.forward(input_tensor, self._label) + loss
        return input_tensor

    def backward(self):
        """
        starting from the loss_layer and propagating it back through the network.
        :return: error_tensor after backpropagation through the network
        """
        error_tensor = self.loss_layer.backward(self._label)
        for each_layer in self.layers[::-1]:
            error_tensor = each_layer.backward(error_tensor)

        return error_tensor

    def append_layer(self, layer):
        """
        If the layer is trainable, make a deepcopy and set it for the layer
        :param layer: a trainable or non-trainable layer
        :return: list of layers which are trainable and after deepcopy of the neural network’s optimizer
        """
        if layer.trainable:
            layer.optimizer = deepcopy(self.optimizer)
            layer.initialize(self.weight_initializer, self.bias_initializer)
        self.layers.append(layer)
        pass


    def train(self, iterations):
        """
        trains the network for iterations and stores the loss for each iteration.
        :param iterations: times of iterations
        :return:
        """
        for i in range(iterations):
            self._phase = 1
            self.loss.append(np.sum(np.square(self.forward())))
            self.backward()
        pass

    def test(self, input_tensor):
        """
        :param input_tensor: input_tensor which is propagated through the network
        :return: the prediction of the last layer
        """
        self._phase = 2
        for each_layer in self.layers:
            each_layer.testing_phase = True
            input_tensor = each_layer.forward(input_tensor)
        return input_tensor
        pass

    @property
    def phase(self):
        return self._phase


def save(filename, net):
    # save object net in the opened file
    file = open(filename, 'wb')
    pickle.dump(net, file)
    file.close()


def load(filename, data_layer):
    # read obj in the opened file
    file = open(filename, 'rb')
    network = pickle.load(file)
    file.close()
    network.data_layer = data_layer
    return network
