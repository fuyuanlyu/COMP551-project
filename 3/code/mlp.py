from convnet import trainloader,testloader,classes
from util import *
from layer import *
import numpy as np 

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
        self.train_fitted = np.array([])

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
        derv_cost_by_activation = -cross_entropy(self.data_Y,np.argmax(self.net[self.output_layer_id].a,axis=0))
        derv_activation_by_summation = sigmoid_derivative(self.net[self.output_layer_id].a)
        self.net[self.output_layer_id].d = np.multiply(derv_cost_by_activation,derv_activation_by_summation)

        # Hidden layer
        for i in np.arange(1,self.output_layer_id,dtype=int)[::-1]:
            d_next = self.net[i+1].d
            if self.net[i+1].bias:
                d_next = d_next[1:]
            derv_summation_lnext_by_activation = self.net[i+1].W.transpose().dot(d_next)
            derv_activation_by_summation = sigmoid_derivative(self.net[i].a)
            self.net[i].d = np.multiply(derv_summation_lnext_by_activation,derv_activation_by_summation)
    
    def compute_gradients_errors(self):
        #Update layer errors d
        self.back_propgation_error()
        # Hidden layer gradients
        for i in np.arange(1,self.output_layer_id+1,dtype=int)[::-1]:
            layer_cur_activations = self.net[i-1].a
            layer_next_errors = self.net[i].d
            if self.net[i].bias:
                layer_next_errors = layer_next_errors[1:] #remove bias error row of layer_next
            
            self.net[i].g = layer_next_errors.dot(layer_cur_activations.transpose())
            # Stochastic gradient descent: normalize the gradient by batch size
            self.net[i].g = self.net[i].g / self.batch_size

    def update_weights(self,lr):
        #Get gradient error
        self.compute_gradients_errors()
        for i in np.arange(0,self.output_layer_id,dtype=int)[::-1]:
            self.net[i].update_weights(lr)
    
    def cost_register(self):
        # Cross entropy loss from sklean
        loss = cross_entropy(self.data_Y,np.argmax(self.net[self.output_layer_id].a,axis=0))
        self.loss.append(loss)

    def get_accurate_number(self,true,prediction):
        # Get multiclass classification accurate number, remember to mean in the end
        return true == prediction

    def feed_forward(self):
        for i in np.arange(0,self.output_layer_id+1, dtype=int):
            if i == 0:
                # This is input layer
                self.net[i].a[1:]= self.data_X.transpose() # a[0] is for bias term
            else:
                self.net[i].z = self.net[i].W.dot(self.net[i-1].a)
                self.net[i].set_activation()
    
    def train_fit(self):
        for epoch in np.arange(self.epochs):
            for i,data in enumerate(trainloader,0):
                self.data_X, self.data_Y = data
                self.data_X = self.data_X.reshape(self.batch_size,-1) #Flatten input into 4x3072
                self.data_X = self.data_X.numpy()
                self.data_Y = self.data_Y.numpy()

                self.feed_forward()
                self.cost_register()

                #Fitting
                batch_predictions = self.predict(self.data_X,in_training=True) # Void input
                self.train_fitted = np.append(self.train_fitted,batch_predictions)
                self.training_acc = np.append(self.training_acc,self.get_accurate_number(self.data_Y,batch_predictions))

                self.update_weights(self.lr)
                if i % 2000 == 1999:    # print every 2000 mini-batches average loss
                    print('[%d, %5d] average loss: %.3f and training accuracy: %.3f'%(epoch + 1, i + 1, np.sum(self.loss[-1999:-1]) / 2000, np.mean(self.training_acc[-1999:-1])))
    
    def predict(self,x_input,in_training=False):
        '''
        Input should be fit to unflattened images PyTorch dataloader
        '''
        if not in_training:
            self.data_X = x_input
            self.data_X = self.data_X.numpy()
            self.feed_forward()
        prediction = np.argmax(self.net[self.output_layer_id].a,axis=0) # return the most possible labels
        return prediction
    
if __name__ == '__main__':
    # For debug
    mlp = MLP(epochs=100,lr=1e-4)
    mlp.train_fit()




    