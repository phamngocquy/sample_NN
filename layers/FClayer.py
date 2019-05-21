from layers.layer import Layer
import numpy as np


class FClayer(Layer):

    def __init__(self, input_shape, output_shape):
        """

        :param input_shape: 1x3
        :param output_shape: 1x4
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = np.random.rand(input_shape[1], output_shape[1]) - 0.5
        self.bias = np.random.rand(1, output_shape[1]) - 0.5

    def input(self):
        pass

    def output(self):
        pass

    def input_shape(self):
        pass

    def output_shape(self):
        pass

    def forward_propagation(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        current_layer_error = np.dot(output_error, self.weights.T)
        