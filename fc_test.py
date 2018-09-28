import numpy as np

# fc layer implementation
class FC(object):
    def __init__(self, num_in, num_out):
        self.num_in = num_in
        self.num_out = num_out
        self.weight = np.random.rand(num_in, num_out)
        self.bias = np.zeros((1, num_out), dtype=float)

    def forward(self, data):
        self.data = data
        self.output = np.dot(self.data, self.weight) + self.bias
        return self.output

    def backward(self, gradient_wrt_output):
        self.gradient_wrt_weight = np.dot(self.data.T, gradient_wrt_output) 
        self.gradient_wrt_bias = gradient_wrt_output.sum(axis=0).reshape(1,-1)
        self.gradient_wrt_input = np.dot(gradient_wrt_output, self.weight.T)
        return self.gradient_wrt_input

    def update(self, lr):
        self.weight = self.weight - lr*self.gradient_wrt_weight
        self.bias = self.bias - lr*self.gradient_wrt_bias

class MSELoss(object):
    def __init__(self):
        pass
    
    def forward(self, data, gt):
        self.data = data
        self.gt = gt
        self.n_data_elements = reduce(lambda x, y:x*y, data.shape)
        self.output = ((self.data-self.gt)**2).mean()
        return self.output
    
    def backward(self):
        self.gradient_wrt_input = 2*(self.data-self.gt)/self.n_data_elements
        return self.gradient_wrt_input

class Sigmoid(object):
    def __init__(self):
        pass

    def forward(self, data):
        self.data = data
        self.output = 1./(1+np.exp(-self.data))
        return self.output

    def backward(self, gradient_wrt_output):
        self.gradient_wrt_input = gradient_wrt_output * self.output * (1. - self.output)
        return self.gradient_wrt_input


data = np.linspace(-1, 1, 100).reshape(-1,1)
y = data*4 + 0.9213 #+ np.random.randn(100,1)*0.5
fc1 = FC(1,1)

loss = MSELoss()
for i in range(20000):
    o1 = fc1.forward(data)
    o2 = loss.forward(o1, y)
    print('{}'.format(o2))

    gradient_wrt_o1 = loss.backward()
    #print gradient_wrt_o2_act
    #print gradient_wrt_o2
    gradient_wrt_data = fc1.backward(gradient_wrt_o1)
    #print gradient_wrt_o1_act
    #print gradient_wrt_o1

    fc1.update(0.1)
    print fc1.weight
    print fc1.bias

