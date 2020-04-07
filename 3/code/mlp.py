from convnet import trainloader,testloader,classes
from util import *
from layer import *
import numpy as np 
from matplotlib import pyplot as plt

class MLP:
    '''
    Multilayer perceptron

    input:
    ---------
    -hidden_layer_params: the parameters of hidden layers
    -train_data: input iterable data loader
    -test_data: input iterable data loader
    -lr: learning rate
    -epochs
    '''

    def __init__(self,hidden_layer_params=(120,84),train_data=trainloader,test_data=testloader,lr=1e-3,epochs=1):
        self.hidden_layer_params,self.train_data,self.test_data = hidden_layer_params,train_data,test_data

        self.lr = lr # Learning rate
        self.epochs = epochs
        # Construct network
        self.net = self.build_net()
        self.output_layer_id = self.net[-1].layer_id

        self.data_X = None # Train batch data
        self.data_Y = None # Train batch labels
        self.loss = [] # Cross entropy loss
        self.training_acc = []
        self.train_fitted = []

    def build_net(self):
        self.batch_size = self.train_data.batch_size
        Layer.batch_size = self.batch_size
        net = []

        # Input layer
        h,w,c = self.train_data.sampler.data_source.data.shape[1:] # height, width and channel size of each data sample
        new_layer = input_layer(h*w*c,bias=True,layer_id=0)
        net.append(new_layer)

        # Hidden layers
        for i in range(len(self.hidden_layer_params)):
            if i== 0:
                new_layer = hidden_layer(h*w*c,self.hidden_layer_params[i],bias=True,layer_id=i+1)
            else:
                new_layer = hidden_layer(self.hidden_layer_params[i-1],self.hidden_layer_params[i],bias=True,layer_id=i+1)
            net.append(new_layer)

        # Output layer
        new_layer = output_layer(self.hidden_layer_params[-1],len(classes),layer_id=len(self.hidden_layer_params)+1)
        net.append(new_layer)

        [l.print_layer() for l in net]
        return net
                
    def back_propgation_error(self):
        '''
        Implement back propagation to compute backward error. 
        There are two-stage process:
            1. Computation to get the output layer error
            2. Computation to get the hidden layers errors
        Update layers.d
        '''
        # Output layer batch bp
        derv_cost_by_activation = np.sum( -one_hot(self.data_Y) + self.net[self.output_layer_id].output ,axis=1 ) # Should be the derv of softmax?
        derv_activation_by_summation = self.net[self.output_layer_id].get_derivative_of_activation()
        self.net[self.output_layer_id].d = np.multiply(derv_cost_by_activation, derv_activation_by_summation)

        # Hidden layer
        for i in np.arange(1,self.output_layer_id,dtype=int)[::-1]:
            d_next = self.net[i+1].d
            if self.net[i+1].bias:
                d_next = d_next[1:]
            derv_summation_lnext_by_activation = self.net[i].W.transpose().dot(d_next)
            derv_activation_by_summation = self.net[i].get_derivative_of_activation()
            self.net[i].d = np.multiply(derv_summation_lnext_by_activation,derv_activation_by_summation)
    
    def compute_gradients_errors(self,layer):
        #Update layer errors d

        # Output layer gradients
        if layer == self.output_layer_id:
            layer_cur_activations = self.net[self.output_layer_id].a 
            layer_next_errors = -one_hot(self.data_Y) + self.net[self.output_layer_id].output 
            self.net[self.output_layer_id].g = 0.9*self.net[self.output_layer_id].g + (1-0.9)*layer_next_errors.transpose().dot(layer_cur_activations.transpose())/self.batch_size # Stochastic gradient descent with momentum
        else:
            # Hidden layer gradients
            layer_cur_activations = self.net[layer].a
            layer_next_errors = self.net[layer+1].d
            if self.net[layer+1].bias:
                layer_next_errors = layer_next_errors[1:] #remove bias error row of layer_next
            
            self.net[layer].g = 0.9*self.net[layer].g + (1-0.9)*layer_next_errors.dot(layer_cur_activations.transpose())/self.batch_size

    def update_weights(self,lr,eps=1e-2):
        # Get gradient error, implement SGD
        self.back_propgation_error()
        
        for i in np.arange(1,self.output_layer_id+1,dtype=int)[::-1]:
            # No minibatch, but can use it to improve            
            self.compute_gradients_errors(i)
            self.net[i].update_weights(lr)
    
    def cost_register(self):
        # Cross entropy loss from sklean
        return softmax_cross_entropy(self.net[self.output_layer_id].output,self.data_Y)/(self.batch_size-1)

    def get_accurate_number(self,true,prediction):
        # Get multiclass classification accurate number, remember to mean in the end
        return true == prediction

    def feed_forward(self):
        for i in np.arange(0,self.output_layer_id+1, dtype=int):
            if i == 0:
                # This is input layer
                self.net[i].a[1:]= self.data_X.transpose() # a[0] is for bias term
            elif i == 1:
                self.net[i].a = self.net[i-1].a
            else:
                self.net[i].z = self.net[i-1].W.dot(self.net[i-1].a)
                self.net[i].set_activation()
    
    def train_fit(self):
        # eps = 5e-5 # For early stop but need to fit all the data
        for epoch in np.arange(self.epochs):
            loss = []
            train_fitted = []
            training_acc = []
            for i,data in enumerate(trainloader,0):
                self.data_X, self.data_Y = data
                self.data_X = self.data_X.reshape(self.batch_size,-1) #Flatten input into 4x3072
                self.data_X = self.data_X.numpy()
                self.data_Y = self.data_Y.numpy()

                self.feed_forward()
                loss.append(self.cost_register())

                #Fitting
                batch_predictions = self.predict(self.data_X,in_training=True) # Void input
                train_fitted.append(batch_predictions)
                training_acc.append(self.get_accurate_number(self.data_Y,batch_predictions))

                self.update_weights(self.lr)
                if i % 100 == 99:    # print every 2000 mini-batches average loss
                    print('[%d, %5d] average loss: %.3f and training accuracy: %.3f'%(epoch + 1, i + 1, np.mean(loss[-100*self.batch_size-1:-1]), np.mean(training_acc[-100*self.batch_size-1:-1])))
            self.loss.append(loss)
            self.train_fitted.append(train_fitted)
            self.training_acc.append(training_acc)
        self.plot_debug()

    def plot_debug(self):
        plt.plot(self.loss)
        #plt.savefig('test_lr{:}.png'.format(self.lr))

    def predict(self,x_input,in_training=False):
        '''
        Input should be fit to unflattened images PyTorch dataloader
        '''
        if not in_training:
            self.data_X = x_input
            self.feed_forward()
        prediction = np.argmax(self.net[self.output_layer_id].output,axis=1) # return the most possible labels
        return prediction
    
    def test(self):
        TP = []
        for _,data in enumerate(testloader,0):
            inputs, labels = data
            inputs = inputs.reshape(self.batch_size,-1) #Flatten input into 4x3072
            inputs = inputs.numpy()
            labels = labels.numpy()

            predicted = self.predict(inputs)
            TP.append(self.get_accurate_number(labels,predicted))
        print("Test acc: ", np.mean(TP))

    def main(self):
        self.train_fit()
        self.test()
    
if __name__ == '__main__':
    # For debug
    mlp = MLP(hidden_layer_params=(120, 84),epochs=3,lr=0.001)
    mlp.main()