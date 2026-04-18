from pyresdmd.dicts.nn.relu import ReLUModule
from pyresdmd.dicts import BaseDictionary
import torch

class ReLUDictionary(BaseDictionary):
    def __init__(self, input_dim: int, n_functions: int, hidden_dim: int = 32, hidden_layers: int = 1):
        '''
        Initializes a neural network dictionary.
        
        Parameters
        ----------------------------------
        input_dim : int 
            Input dimension for dictionary. 
        
        n_functions : int 
            Number of trainable dictionary functions. 
        
        hidden_dim : int 
            Default 32. Number of dimensions to be used in hidden layers. 
        
        hidden_layers : int 
            Default 1. Number of hidden layers between the input layer and output layer. 
        
        Returns 
        ----------------------------------
        Nothing, sets class attributes.
        '''
        super().__init__()
        self._input_dim = input_dim 
        self._n_functions = n_functions 
    
        self.networks = nn.ModuleList([
            NeuralNetworkModule(input_dim, hidden_dim, hidden_layers) 
            for _ in range(n_functions)
        ])
    
    @property 
    def size(self) -> int:
        '''
        Returns the size of the dictionary.
        '''
        return self._n_functions 
    
    @property
    def input_dim(self) -> int:
        '''
        Returns the common input dimension of the dictionary functions.
        '''
        return self._input_dim 
    
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Performs a forward pass.
        '''
        return torch.stack([net(x) for net in self.networks], dim=1)
