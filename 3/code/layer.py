import numpy as np
from util import *
class Layer:
    '''
    Define a fully connected layer
    
    Input:
    ---------
    n_units_input: int
    n_units_output: int
    bias: bool
    layer_id: int
    '''
    batch_size = None
    def __init__(self,n_units_input,n_units_output,bias,layer_id):
        self.n_units_input,self.n_units_output,self.bias,self.layer_id = n_units_input,n_units_output,bias,layer_id

        # Initialization
        # Summation vector
        self.z = self.init_vector((self.n_units_input ,Layer.batch_size))
        # Activated summation vector
        self.a = self.init_vector((self.n_units_input ,Layer.batch_size))
        self.set_activation()
        # Weight matrix from current layer to next layer
        self.W = self.init_weights()
        # Delta-error vector
        self.d = self.init_vector((self.n_units_input + self.bias, Layer.batch_size))
        # Gradient error vector
        self.g = self.init_vector(self.W.shape)


    def init_weights(self):
        if self.n_units_output is None:
            return np.array([])
        else:
            weights = np.random.randn(self.n_units_output*(self.n_units_input+ self.bias))
            weights = weights.reshape(self.n_units_output,self.n_units_input+ self.bias)
            return weights
        
    def init_vector(self,n_dim):
        if n_dim[0] is None:
            return np.array([])
        else:
            return np.random.normal(size=n_dim)

    def set_activation(self):
        self.a = ReLu(self.z)
        if self.bias:
            self.add_activation_bias()

    def add_activation_bias(self):
        if len(self.a.shape) == 1:
            self.a = np.vstack((1, self.a))
        else:
            self.a = np.vstack((np.ones(self.a.shape[1]), self.a))
    
    def get_derivative_of_activation(self):
        return ReLu_derivative(self.a)

    def update_weights(self,lr):
        self.W -= (lr*self.g)
    
    def print_layer(self):
        #print("W:\n {} \n".format(self.W))
        #print("z: {}".format(self.z))
        #print("a: {}".format(self.a))
        #print("g: {}".format(self.g))
        print("Layer {}: ({},{})".format(self.layer_id,self.n_units_input,self.n_units_output))

# Define common used layers using the defined parent class
class input_layer(Layer):
    def __init__(self,n_units_input, n_units_output=None, bias=True, layer_id=0):
        Layer.__init__(self, n_units_input, n_units_output, bias, layer_id)
        self.z = np.array([])
        #self.d = self.init_vector((self.n_units_input + 1, Layer.batch_size))

class hidden_layer(Layer):
    def __init__(self,n_units_input, n_units_output, bias=True, layer_id=None):
        Layer.__init__(self, n_units_input, n_units_output, bias, layer_id)

class output_layer(Layer):
    def __init__(self,n_units_input,n_units_output,layer_id):
        Layer.__init__(self, n_units_input, n_units_output, bias=False, layer_id =layer_id)
        # self.g = np.array([])

    def set_activation(self):
        # Softmax in the output layer
        self.a = ReLu(self.z)

    def init_weights(self):
        weights = np.random.randn(self.n_units_output*(self.n_units_input))
        weights = weights.reshape(self.n_units_output,self.n_units_input)
        return weights
    
    @property
    def output(self):
        return softmax(self.W.dot(self.a).T)

if __name__ == '__main__':
    # For debug
    Layer.batch_size = 4
    l = Layer(120,80,True,0)
    l.print_layer()