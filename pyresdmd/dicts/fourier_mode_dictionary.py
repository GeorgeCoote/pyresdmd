from pyresdmd.dicts.dictionary import Dictionary
import torch

class FourierModeDictionary(Dictionary):
    def __init__(self, max_mode : int) -> None:
        '''
        Initializes Fourier mode dictionary. 
        
        Parameters 
        -------------------------
        max_mode : int 
            If n = max_mode: use 1, sin(pi*x), cos(pi*x), ..., sin(n*pi*x), cos(n*pi*x)
        
        Returns 
        -------------------------
        None, just sets class attributes
        '''
        super().__init__()
        self.max_mode = max_mode 
    
    @property 
    def size(self) -> int:
        '''
        Returns the size of the dictionary.
        '''
        return 2*self.max_mode + 1
    
    @property 
    def input_dim(self) -> int:
        '''
        Returns the shared input dimension of the dictionary functions.
        '''
        return 1
    
    def evaluate(self, x : torch.Tensor) -> torch.Tensor:
        '''
        Evaluates the dictionary at the point x. 
        
        Parameters 
        -------------------------
        x : torch.Tensor 
            The point at which to evaluate. 
        
        Returns 
        -------------------------
        torch.tensor 
            A tensor which represents every dictionary function evaluated at x.
        '''
        x = x.flatten()
        
        modes = torch.arange(1, self.max_mode + 1, dtype = x.dtype, device = x.device)
        args = torch.pi * modes[None, :] * x[:, None]
        
        return torch.cat([
            torch.ones(x.shape[0], 1, dtype = x.dtype, device = x.device),
            torch.sin(args),
            torch.cos(args)
        ], dim = 1)
    
    def __repr__(self) -> str:
        '''
        Prints a debug representation of a Fourier mode dictionary, reporting the maximum mode.
        '''
        return f"FourierModeDictionary(max_mode = {self.max_mode})"