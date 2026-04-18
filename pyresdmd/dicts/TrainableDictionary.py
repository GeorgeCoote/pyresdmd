import torch
import torch.nn as nn
from pyresdmd.dicts import BaseDictionary
from pyresdmd.compute.spectra import spectra

class TrainableDictionary(nn.Module):
    '''
    Implements a trainable dictionary, using ResDMD residuals as a loss.
    '''
    def __init__(self, dictionary : BaseDictionary, quadrature_weights : torch.Tensor = None, eps : float = 1e-8) -> None:
        '''
        Sets up trainable dictionary. 

        Parameters
        ----------------------------------
        dictionary : BaseDictionary 
            Initialized dictionary to train
        
        quadrature_weights : torch.Tensor
            Quadrature weights to use for EDMD computation. 
        
        eps : float 
            Epsilon offset to use in condition number computation. 
        
        Returns 
        ----------------------------------
        None, sets class attributes.
        '''
        super().__init__()
        self.dictionary = dictionary 
        self.dictionary_size = dictionary.size 
        self.eps = eps
        self.register_buffer('quadrature_weights', quadrature_weights)
    
    @property 
    def size(self) -> int:
        return self.dictionary_size
        
    @property 
    def input_dim(self) -> int:
        return self.dictionary.input_dim 
    
    def evaluate(self, x : torch.Tensor) -> torch.Tensor:
        return self.dictionary.evaluate(x)
    
    def get_eps(self):
        '''
        Gets epsilon offset. 
        '''
        return self.eps 
    
    def set_eps(self, eps : float = 1e-8):
        '''
        Sets epsilon offset. Default value 1e-8. 
        '''
        self.eps = eps
    
    def compute_psi(self, x : torch.Tensor, y : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Produces Hankel matrices for the trained matrix. 

        Parameters 
        ----------------------------------
        x : torch.Tensor 
            Trajectory data for x 
        
        y : torch.Tensor 
            Trajectory data for x 
        
        Returns 
        ----------------------------------
        tuple[torch.Tensor, torch.Tensor] 
            Produces a tuple containing the two Hankel matrices Psi_X and Psi_Y. 
        '''
        return self.evaluate(x), self.evaluate(y)
    
    def forward(self, x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
        '''
        Performs a forward pass using eigvals_eigvecs and the trained dictionary. 
        
        Parameters 
        ----------------------------------
        x : torch.Tensor 
            Trajectory data for x 
        
        y : torch.Tensor 
            Trajectory data for x 
        
        Returns 
        ----------------------------------
        tuple[torch.Tensor, torch.Tensor] 
            Result of forward pass
        '''
        Psi_X, Psi_Y = self.compute_psi(x, y) # trained dictionary comes in here
        return spectra(Psi_X, Psi_Y, self.quadrature_weights, self.eps)
