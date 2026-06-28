import torch
import torch.nn as nn 

class ReLUModule(nn.Module):
    '''Implements a ReLU module'''
    def __init__(self, input_dim : int, hidden_dim : int = 32, hidden_layers : int = 1) -> None:
        '''
        Initializes a neural network with ReLU 
        
        Parameters
        ----------------------------------
        input_dim : int
            Number of dimensions of input. Denoted d in Colbrook.
        
        hidden_dim : int
            Dimension to use for hidden layers between input and output.

            Default 32

        hidden_layers : int
            Number of hidden layers, where a layer consists of a linear function and the ReLU activation function

            Default 1
        
        Returns
        ----------------------------------
        Nothing, only sets class attributes.
        '''
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim 
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        
        for i in range(hidden_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        '''
        Function to perform forward iteration. 
        
        Parameters 
        ----------------------------------
        x
            Snapshot data provided as a tensor of size (M, input_dim) where M is the number of snapshots.
        
        Returns
        ----------------------------------
        torch.tensor
            Tensor of shape (M, ) representing the dictionary function value. 
        '''
        return self.net(x).squeeze(-1) # self.net(x) returns tensor of size (N, 1), remove last dim.
