import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        """

        :param prediction_tensor: predict label, yk_hat
        :param label_tensor: true label, yk
        :return: loss
        """
        self.prediction_tensor = prediction_tensor
        #machine limits for floating point types.
        eps = np.finfo(float).eps

        #the Cross Entropy Loss requires predictions to be greater than 0
        pred = np.copy(prediction_tensor)
        pred[np.where(label_tensor != 1)] = 0

        #loss = sum(-ln(yk_hat + eps))
        loss = np.sum(-np.log(np.sum(pred, axis=1) + eps))

        return loss

    def backward(self, label_tensor):
        """

        :param label_tensor: true label, yk
        :return: Error(n) = -y / (y_hat + eps)
        """

        return -label_tensor / (self.prediction_tensor + np.finfo(float).eps)

