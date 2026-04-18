import torch 
from pyresdmd.dicts import BaseDictionary
from pyresdmd.dicts.nn.siren import SIRENModule

class SIRENDictionary(BaseDictionary):
    def __init__(self, input_dim : int, n_functions : int, hidden_dim : int, hidden_layers : int = 2, w0_first : float = 30.0, w0 : float = 1.0):
        '''
        Initializes trainable SIREN dictionary.
        
        Parameters
        ----------------------------------
        input_dim : int 
            Common input dimension for dictionary functions. 
        
        n_functions : int 
            Size of dictionary. 
        
        hidden_dim : int 
            Dimension of hidden layers. 
        
        hidden_layers : int
            Number of hidden layers. 
        
        w0_first : float 
            Frequency of first sine layer. 
        
        w0 : float 
            Frequency of other sine layers. 
        
        Returns 
        ----------------------------------
        Nothing, sets class attributes.
        '''
        super().__init__()
        self._input_dim = input_dim 
        self._n_functions = n_functions
        self.hidden_dim = hidden_dim
        
        self.networks = nn.ModuleList([
            SIRENModule(input_dim, hidden_dim, hidden_layers, w0_first, w0)
            for _ in range(n_functions)
        ])
        
    @property
    def size(self) -> int:
        '''
        Returns size of dictionary. 
        '''
        return self._n_functions 
    
    @property
    def input_dim(self) -> int:
        '''
        Returns common input dimension of dictionary functions. 
        '''
        return self._input_dim 
    
    def evaluate(self, x : torch.Tensor) -> torch.Tensor:
        '''
        Evaluates the neural network at a point
        '''
        return torch.stack([net(x) for net in self.networks], dim = 1)
