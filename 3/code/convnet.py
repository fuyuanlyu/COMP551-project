import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# Global Variables: Load and preprocess the data
def transform_method(method='normalize'):
    if (method=='random_flip_crop_padding'):
            transform = transforms.Compose(
                            [transforms.RandomHorizontalFlip(),
                             transforms.RandomCrop(size=[32,32], padding=[0, 2, 3, 4]),
                             transforms.ToTensor()])
    if (method =='normalize'):
            transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if (method =='random_change_brightness'):
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(20),
                transforms.ToTensor()])

    return transform

transforms_train=transform_method(method='normalize')
transforms_test=transform_method(method='normalize')
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transforms_train,target_transform=None) ##you can select which transform to use

trainloader = torch.utils.data.DataLoader(trainset,batch_size=16,shuffle=True,num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transforms_test) ##it seem that test set doesn't need transform
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Global function if you want to visualize the image
def imshow(img):
    '''
    Input: image data in the format of torch.tensor
    '''
    img = img / 2 + 0.5     # Unnormalize in the same method in transforms
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class ConvNet(nn.Module):
    '''
    Input:
    ----------
    n_convs: int, n conv layers, the pooling layer size is fixed and the conv+pooling is considered as one conv layer. Note that the last conv layer has no pooling. 

    convs_params: tuples, if you want to customize the conv layer numbers, please specify the (input_channel,output_channel,kernel_size) as a (n_convs x 3) tuple. Note that the output_channel size has to be same with the next input_channel size!

    n_fc: int, n fully connected layers, because the output layer will be another fc layer, so there are actually n_fc+1 fully connected layers

    fc_params: tuples, if you want to customize the hidden layer sizes of each n_fc layers, please specify the (hidden layer sizes) as a (n_fc,1) tuple.

    optimizer: str, 'SGD' or 'Adam'
    
    '''
    def __init__(self,n_convs=2,convs_params=((3,6,5),(6,16,5)),n_fc=2,fc_params=(120,84),optimizer='SGD',epoch=1):
        super(ConvNet,self).__init__()
        self.n_convs,self.convs_params,self.n_fc,self.fc_params,self.optim,self.epoch = n_convs,convs_params,n_fc,fc_params,optimizer,epoch

        # Pooling
        self.pool = nn.MaxPool2d(2,2) # Kenrl size; stride
        # Define conv layers
        self.conv_list = nn.ModuleList()
        for i_conv in range(self.n_convs):
            self.conv_list.append(nn.Conv2d(self.convs_params[i_conv][0],self.convs_params[i_conv][1],self.convs_params[i_conv][2]))
        # Define fc layers
        self.fc_list = nn.ModuleList()
        for i_fc in range(n_fc):
            if i_fc == 0:
                self.fc_list.append(nn.Linear(self.convs_params[-1][1]*self.convs_params[-1][2]*self.convs_params[-1][2],self.fc_params[i_fc]))
            else:
                self.fc_list.append(nn.Linear(
                    elf.fc_params[i_fc-1],self.fc_params[i_fc]))
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
        
        # Convolution
        for conv_layer in self.conv_list:
            x = self.pool(conv_layer(x))     
        # Flatten
        x = x.view(-1,self.convs_params[-1][1]*self.convs_params[-1][2]*self.convs_params[-1][2])
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
        if i % 2000 == 1999:    # print every 2000 mini-batches average loss
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

def test(PATH):
    '''
    Test accuracy for a saved trained model
    '''
    net=torch.load(PATH)
    net.eval()
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

def main(net,PATH):
    '''
    The main function to run
    '''
    for epoch in range(net.epoch):
        train(net,epoch)
    torch.save(net, PATH)
    accuracy = test(PATH)
    print('Model accuracy: %.3f' % accuracy)

if __name__ == '__main__':
    PATH = 'saved_models/cnn_test.pth' # Saved path
    net = ConvNet() # Define you parameters for ConvNet here
    print('ConvNet structure: \n',net)
    main(net,PATH)
