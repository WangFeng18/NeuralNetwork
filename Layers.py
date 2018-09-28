import numpy as np
from itertools import product as cross
from function import conv, rotate, pad, rearange, trans_coord_from_1d_to_2d

# fc layer implementation
class Layer(object):
    def update(self, lr):
        for i in range(len(self.parameters)):
            self.parameters[i] -= lr * self.gradient_wrt_parameters[i]

           
class Convolution(Layer):
    # this version just support stride=1 without paddings.
    def __init__(self, n_filter, kernel_size, input_size, input_channel):
        self.kernel_size = kernel_size
        self.parameters = []
        self.n_filter = n_filter
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.input_channel = input_channel
        weight = np.random.rand(n_filter, kernel_size, kernel_size, input_channel)
        bias = np.zeros((1, input_size-kernel_size+1, input_size-kernel_size+1, n_filter), dtype=float)
        self.parameters.append(weight)
        self.parameters.append(bias)
    
    def forward(self, data):
        # data must be bs * input_size * input_size * channel
        assert data.shape[1:] == (self.input_size, self.input_size, self.input_channel)
        self.output = conv(data, self.parameters[0]) + self.parameters[1]
        return self.output
        
    def backward(self, gradient_wrt_output):
        weight = self.paramters[0]
        paddings = [weight.shape[0]-1, weight.shape[1]-1]
        self.gradient_wrt_parameters = []
        gradient_wrt_input = gradient_wrt_output * conv(pad(self.output, paddings), rearange(rotate(weight)))
        gradient_wrt_weight = rearange( conv(rearange(self.data), rearange(gradient_wrt_output)) )
        gradient_wrt_bias = gradient_wrt_output.sum(axis=0).reshape(self.parameters[1].shape)
        self.gradient_wrt_parameters.append(gradient_wrt_weight)
        self.gradient_wrt_parameters.append(gradient_wrt_bias)
        return gradient_wrt_input

class MaxPooling(Layer):
    def __init__(self, kernel_size, stride=1, valid=True):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, data):
        self.data = data
        mH = (data.shape[1]-self.kernel_size[0])/self.stride
        mW = (data.shape[2]-self.kernel_size[1])/self.stride
        output = np.zeros((data.shape[0], mH, mW, data.shape[3]), dtype=float)
        max_position = np.zeros([data.shape[0], mH, mW, data.shape[3], 2]), dtype=float)
        for mh,mw in cross(range(mH), range(mW)):
            block = data[:,mh*self.stride:mh*self.stride+self.kernel_size[0]-1, \
                        mw*self.stride:mw*self.stride+self.kernel_size[1]-1,:]
            sp = block.shape
            block = block.reshape([sp[0], sp[1]*sp[2], sp[3]])
            output[:,mh,mw,:] = block.max(axis=1)
            max_index_1d = block.argmax(axis=1)
            max_index_2d = trans_coord_from_1d_to_2d(max_index_1d, sp[2])
            max_position[:,mh,mw,:,:] = max_index_2d
        self.output = output
        self.max_position = max_position
        return output

    def backward(self, gradient_wrt_output):
        mH = (self.data.shape[1]-self.kernel_size[0])/self.stride
        mW = (self.data.shape[2]-self.kernel_size[1])/self.stride
        gradient_wrt_input = np.zeros_like(self.data, dtype=float)
        for mh,mw in cross(range(mH), range(mW)):
            block = gradient_wrt_input[:,mh*self.stride:mh*self.stride+self.kernel_size[0]-1, \
                                    mw*self.stride:mw*self.stride+self.kernel_size[1]-1,:]
            arg_max_h = mh*self.stride + self.max_position[:,mh,mw,:,0]
            arg_max_w = mw*self.stride + self.max_position[:,mh,mw,:,1]
            for i_batch, i_channel in cross(range(self.data.shape[0]), range(self.data.shape[3])):
                gradient_wrt_input[i_batch,arg_max_h[i_batch, i_channel],argmax_w[i_batch,i_channel],i_channel] += \
                        gradient_wrt_output[i_batch,mh,mw,i_channel]
        self.gradient_wrt_input = gradient_wrt_input
        return gradient_wrt_input


class FC(Layer):
    def __init__(self, num_in, num_out):
        self.num_in = num_in
        self.num_out = num_out
        self.parameters = []
        weight = np.random.rand(num_in, num_out)
        bias = np.zeros((1, num_out), dtype=float)
        self.parameters.append(weight)
        self.parameters.append(bias)

    def forward(self, data):
        self.data = data
        self.output = np.dot(self.data, self.parameters[0]) + self.parameters[1]
        return self.output

    def backward(self, gradient_wrt_output):
        self.gradient_wrt_parameters = []
        self.gradient_wrt_parameters.append(np.dot(self.data.T, gradient_wrt_output))
        self.gradient_wrt_parameters.append(gradient_wrt_output.sum(axis=0).reshape(1,-1))
        self.gradient_wrt_input = np.dot(gradient_wrt_output, self.parameters[0].T)
        return self.gradient_wrt_input

    def __str__(self):
        str = ''
        str +=  '--------FC_layer--------\n'
        str += 'weight shape: {}'.format(self.parameters[0].shape)+'\n'
        str += 'bias   shape: {}'.format(self.parameters[1].shape)+'\n'
        str += 'weight      : {}'.format(self.parameters[0])+'\n'
        str += 'bias        : {}'.format(self.parameters[1])+'\n'
        str += '------------------------\n'
        return str

class MSELoss(object):
    def __init__(self):
        pass
    
    def forward(self, data, gt):
        self.data = data
        self.gt = gt
        self.n_data_elements = reduce(lambda x, y:x*y, data.shape)
        self.output = ((self.data-self.gt)**2).mean()
        return self.output
    
    def backward(self, gradient_wrt_output):
        assert gradient_wrt_output is None
        self.gradient_wrt_input = 2*(self.data-self.gt)/self.n_data_elements
        return self.gradient_wrt_input

class ReLU(object):
    def __init__(self):
        pass
    
    def forward(self, data):
        self.data = data
        self.output = np.maximum(self.data, 0)
        return self.output

    def backward(self, gradient_wrt_output):
        self.gradient_wrt_input = gradient_wrt_output * (self.output != np.zeros_like(self.output)).astype(np.float)
        return self.gradient_wrt_input

class Sigmoid(object):
    def __init__(self):
        pass

    def forward(self, data):
        self.data = data
        self.output = 1./(1+np.exp(-self.data))
        #print self.output
        return self.output

    def backward(self, gradient_wrt_output):
        self.gradient_wrt_input = gradient_wrt_output * self.output * (1. - self.output)
        return self.gradient_wrt_input

class Container(object):
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, data_input):
        for layer in self.layers:
            data_input = layer.forward(data_input)
        return data_input

    def backward(self, gradient_wrt_output):
        for layer in self.layers[::-1]:
            gradient_wrt_output = layer.backward(gradient_wrt_output)

    def update(self, lr):
        for layer in self.layers:
            layer.update(lr)

    def __str__(self):
        str = ''
        for layer in self.layers:
            str += layer.__str__()
        return str
 
if __name__ == '__main__':
    data = np.linspace(-1, 1, 100).reshape(-1,1)
    y = data*3 + np.random.randn(100,1)*0.5
    net = Container(FC(1,1), FC(1,1))
    loss = MSELoss()
    lr = 0.1

    for i in range(1000):
        o = net.forward(data)
        l = loss.forward(o, y)
        g = loss.backward(None)
        net.backward(g)
        print('{}'.format(l))
        net.update(lr)

    print net

