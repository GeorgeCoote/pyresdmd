from abc import ABC, abstractmethod 
import torch

class BaseDictionary(ABC, nn.Module):
    '''
    Abstract class serving as a template for dictionary classes.
    '''
    @property
    @abstractmethod 
    def size(self) -> int:
        '''
        Returns the size of a dictionary.
        '''
        ...
    
    @property 
    @abstractmethod 
    def input_dim(self) -> int:
        '''
        Returns the input dimension of a dictionary. Must be uniform across dictionary functions.
        '''
        ...
    
    @abstractmethod 
    def evaluate(self, x : torch.Tensor) -> torch.Tensor: 
        '''
        Computes the dictionary at the point x. 
        '''
        ...
    
    def forward(self, x : torch.Tensor) -> torch.Tensor: 
        '''
        Function for forward step.
        '''
        return self.evaluate(x) 
