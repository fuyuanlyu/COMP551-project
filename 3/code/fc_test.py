import torch
import torchvision
import torchvision.transforms as transforms
from convnet import trainloader,testloader,classes
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")



class MLP(nn.Module):
    '''
    Input:
    ----------
    n_convs: int, n conv layers, the pooling layer size is fixed and the conv+pooling is considered as one conv layer. Note that the last conv layer has no pooling. 

    convs_params: tuples, if you want to customize the conv layer numbers, please specify the (input_channel,output_channel,kernel_size) as a (n_convs x 3) tuple. Note that the output_channel size has to be same with the next input_channel size!

    n_fc: int, n fully connected layers, because the output layer will be another fc layer, so there are actually n_fc+1 fully connected layers

    fc_params: tuples, if you want to customize the hidden layer sizes of each n_fc layers, please specify the (hidden layer sizes) as a (n_fc,1) tuple.

    optimizer: str, 'SGD' or 'Adam'
    
    '''
    def __init__(self,n_fc=2,fc_params=(120,84),optimizer='SGD',epoch=1):
        super(MLP,self).__init__()
        self.n_fc,self.fc_params,self.optim,self.epoch = n_fc,fc_params,optimizer,epoch

        self.batch_size = trainloader.batch_size
        # Input layer
        self.h,self.w,self.c = trainloader.sampler.data_source.data.shape[1:]

        # Define fc layers
        self.fc_list = nn.ModuleList()
        for i_fc in range(n_fc):
            if i_fc == 0:
                self.fc_list.append(nn.Linear(self.h*self.w*self.c,self.fc_params[i_fc]))
            else:
                self.fc_list.append(nn.Linear(self.fc_params[i_fc-1],self.fc_params[i_fc]))
        self.fc_list.append(nn.Linear(self.fc_params[-1],len(classes)))

        # Optimizer and Loss
        if self.optim == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        elif self.optim == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        else:
            raise ValueError('Please specify optimizer from "SGD" and "Adam"')
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,x): 
        # Flatten
        x = x.view(-1,self.h*self.w*self.c)
        # Fully connected layer
        for i,fc_layer in enumerate(self.fc_list):
            if i < self.n_fc:
                x = F.relu(fc_layer(x))
            else:
                x = fc_layer(x) # Output layer

        return x

def train(net,epoch):
    '''
    Train process for each epoch
    '''
    running_loss = 0.0
    # if not data_aug:
    #     print('not using data augmentatinon')
    # else:
    #     input, label = trainset
    #     datagen = setup_data_aug()
    #     datagen.fit(input)
    #     datagen.flow(input,label,batch_size=32)

    for i,data in enumerate(trainloader,0): #enumerate starts from 0
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        net.optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = net.criterion(outputs, labels) # Cross-entropy loss compared with the labels
        loss.backward() # Backward propagation
        net.optimizer.step() # Optimization update
        
        # print statistics
        running_loss += loss.item()
        if i % 500 == 499:    # print every 2000 mini-batches average loss
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 500))
            running_loss = 0.0

def test(net):
    '''
    Test accuracy for a saved trained model
    '''
    Accurate = 0
    Total = 0
    for i,data in enumerate(testloader,0): #enumerate starts from 0
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        TP = labels == predicted
        TP = TP.data.numpy()
        Accurate += np.sum(TP)
        Total += len(inputs)
    return Accurate/Total

def main(net):
    '''
    The main function to run
    '''
    for epoch in range(net.epoch):
        train(net,epoch)
    accuracy = test(net)
    print('Model accuracy: %.3f' % accuracy)

if __name__ == '__main__':
    net = MLP() # Define you parameters for mlp here
    print('MLP structure: \n',net)
    main(net)
