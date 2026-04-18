import torch
import torch.nn as nn

class SirenLayer(nn.Module):
    def __init__(self, input_dim, output_dim, w0 = 30.0, is_first = False):
        '''
        Initializes a SIREN layer.
        
        Parameters 
        ----------------------------------
        input_dim : int 
            Input dimension of dictionary function. 
        
        output_dim : int 
            Output dimension of dictionary function. 
        
        w0 : float 
            Fixed scaling factor corresponding to the frequency of the sine function.
        
        is_first : bool
            Distinguishes the first layer of the network.
        
        Returns 
        ----------------------------------
        Nothing, only sets class attributes.
        '''
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(input_dim, output_dim)
        self._init_weights(is_first)
    
    def _init_weights(self, is_first):
        '''
        Initializes weights. Not intended to be called directly.

        Uses initialization from Sitzmann's paper. https://arxiv.org/pdf/2006.09661 page 5
        '''
        with torch.no_grad():
            if is_first:
                bound = 1.0 / self.linear.in_features 
            else:
                bound = (6.0 / self.linear.in_features) ** 0.5 / self.w0 
            
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)
        
    def forward(self, x):
        '''
        Performs a forward pass.
        '''
        return torch.sin(self.w0 * self.linear(x))

class SIRENModule(nn.Module):
    def __init__(self, input_dim : int, n_functions : int, hidden_dim : int, hidden_layers : int = 1, w0_first : float = 30.0, w0 : float = 1.0):
        '''
        Initializes trainable SIREN network.
        
        Parameters
        ----------------------------------
        input_dim : int 
            Input dimension for network.
        
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
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim
        
        layers = [SirenLayer(input_dim, hidden_dim, w0 = w0_first, is_first = True)]
               + [SirenLayer(hidden_dim, hidden_dim, w0 = w, is_first = False) for _ in range(hidden_layers)]
                
        output_layer = nn.Linear(hidden_dim, 1)
        
        with torch.no_grad(): # initializing output layer
            bound = (6.0 / hidden_dim) ** 0.5 / w0 
            output.weight.uniform_(-bound, bound)
            output.bias.uniform_(-bound, bound)
            
        return nn.Sequential(*layers, output_layer)
    
    def forward(self, x):
        '''
        Performs a forward pass.
        '''
        return torch.sin(self.w0 * self.linear(x)).squeeze(-1)
