import numpy as np

class Sigmoid(object):
    def __init__(self):
        pass

    def forward(data):
        self.data = data
        self.output = 1./(1+np.exp(-self.data))
        return self.output

    def backward(gradient_wrt_output):
        self.gradient_wrt_input = self.output * (1. - self.output)
        return self.gradient_wrt_input

