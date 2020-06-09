# The Code written by Ali Babolhaveji @ 6/1/2020

import sys
import os
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import torch
#from torchsummary import summary
from .efficientnet_pytorch_3d import EfficientNet3D


def init_params(m):
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        m.weight.data=torch.randn(m.weight.size())*.01#Random weight initialisation
        m.bias.data=torch.zeros(m.bias.size())

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.xavier_uniform(m.bias.data)

class my_deep_clag_loss(nn.Module):
    def __init__(self,in_channels=3 ):
        super(my_deep_clag_loss, self).__init__()
        self.backend = EfficientNet3D.from_name("efficientnet-b7", override_params={'num_classes': 2}, in_channels = in_channels)
        
        #print(self.backend)
        print("************Create RNN**************")
        # Create RNN
        input_dim = 256    # input dimension
        hidden_dim = 100  # hidden layer dimension
        layer_dim = 2     # number of hidden layers
        output_dim = 32   # output dimension
        
        self.rnn = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)
 
        
        

        #opt.landmarks = True
        #self.model_landmarks = generate_model(opt)

        for param in self.backend.parameters():
            param.requires_grad = False

        #for param in self.model_landmarks.parameters():
            #param.requires_grad = False

        self.decode_to_Fake_True =  nn.Sequential(
 #           torch.nn.Linear(1280, 1024), #b0
 #           torch.nn.Linear(1280, 1024), #b1
 #           torch.nn.Linear(1408, 1024), #b2
 #           torch.nn.Linear(1536, 1024), #b3
 #           torch.nn.Linear(1792, 1024), #b4
 #           torch.nn.Linear(2048, 1024), #b5
 #           torch.nn.Linear(2304, 1024), #b6
            torch.nn.BatchNorm1d(2560),
            torch.nn.Linear(2560, 512), #b7
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=.25),
    #        torch.nn.Linear(1024, 512+256),
    #        torch.nn.ReLU(inplace=True),
     #       torch.nn.Dropout(p=.25), 
     #       torch.nn.Linear(512+256, 512),
     #       torch.nn.ReLU(inplace=True),
     #       torch.nn.Dropout(p=.25),
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(inplace=True))

         
        self.head=  nn.Sequential(
          #  torch.nn.Dropout(p=.25),
            torch.nn.BatchNorm1d(32+256),
            torch.nn.Linear(32+256, 32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=.25),
            torch.nn.BatchNorm1d(32),
            torch.nn.Linear(32, 2))
            #torch.nn.Sigmoid())

        self.head.apply(weights_init)
        self.decode_to_Fake_True.apply(weights_init)

    def forward(self, x):
        #print('input--------------------' ,x.shape)
        batch_size = x.shape[0]
        x = self.backend(x) 
        x1= x.view(batch_size,-1,256) 
       # print('backend--------------------' ,x.shape  , x1.shape)
        x2 = self.decode_to_Fake_True(x) #256
       # print('decode_to_Fake_True--------------------' ,x.shape)
        x = self.rnn(x1)
        #print('cattt--------------------' ,x.shape ,x2.shape)
        x = torch.cat((x, x2), 1)
        #print('rnn--------------------' ,x.shape)
        x = self.head(x)
        # input_ = x.t()* 1000
        # input_ = input_.long()
        # print('transpose--------------------' ,input_.shape)
        # embeded = self.embedding(input_ )
        # print('embeded--------------------' ,embeded.shape)
        # print('embeded' , embeded.shape)
        # hidden = torch.zeros((256,batch_size,256) ,requires_grad=True)
        # x = self.LSTM (embeded  )
        # print('LSTM' , embeded.shape)
        #print('out--------------------' ,x.shape)
       


        return x
        
        
# Create RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, 
                          nonlinearity='relu')
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros((self.layer_dim, x.size(0), self.hidden_dim),requires_grad=True).cuda()
            
        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :]) 
        return out
    
    
if __name__ == "__main__": 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = my_deep_fake_EfficientNet()
    model.to(device)
    inputs = torch.randn((2, 3, 200, 150, 150)).to(device)
    print(model(inputs).shape)

