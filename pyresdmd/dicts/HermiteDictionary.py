from pyresdmd.dicts import BaseDictionary
import torch

class HermiteDictionary(BaseDictionary):
    def __init__(self, degree : int):
        '''
        Initializes Hermite dictionary. 
      
        Parameters 
        -------------------------
        degree : int 
            Max degree of polynomials used. 
            
            A degree of 2 means that we will use 1, 2x, 4x^2 - 2.
        
        Returns 
        -------------------------
        None, just sets class attributes
        '''
        
        super().__init__()
        self.degree = degree 
    
    @property 
    def size(self) -> int:
        '''
        Returns the size of the dictionary 
        '''
        return self.degree + 1
    
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
            
            If x has shape (M, 1) or (M,), and N = self.degree, then the output will have shape (M, N).
            
            The jth row is equal to H_0(x_j) * e^(-x_j^2), H_1(x_j) * e^(-x_j^2), ..., H_N(x_j) * e^(-x_j)^2. 
        '''
        x = x.flatten()
        out = torch.zeros(x.shape[0], self.degree + 1, dtype = x.dtype, device = x.device) 
        
        out[:, 0] = 1.0
        if self.degree >= 1:
            out[:, 1] = x
        
        for j in range(2, self.degree + 1):
            out[:, j] = 2 * x * out[:, j - 1] - 2 * (j - 1) * out[:, j - 2]
        
        return out * torch.exp(-x*x)[:, None]
    
    def __repr__(self) -> str:
        '''
        Prints a debug representation of a Hermite dictionary, reporting the degree. Typically to be called
        by writing x in the REPL where x is a HermiteDictionary object, or by print(x). 
        '''
        return f"HermiteDictionary(degree = {self.degree})"
