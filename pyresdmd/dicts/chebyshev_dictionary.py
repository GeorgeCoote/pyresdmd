from pyresdmd.dicts.dictionary import Dictionary
import torch

class ChebyshevDictionary(Dictionary):
    def __init__(self, degree : int, scale : float) -> None:
        '''
        Initializes Chebyshev dictionary. 
      
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
        self.scale = scale
    
    @property 
    def size(self) -> int:
        '''
        Returns the size of the dictionary.
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
            
            The jth row is equal to T_0(s * x_j), T_1(s * x_j), ..., T_N(s * x_j), 
            where s = self.scale.

        Attributes 
        -------------------------
        self.scale : float 
            A multiplicative factor applied to the input before evaluation. The 
            Chebyshev recurrence is applied to (scale * x) rather than to x directly, 
            so each column j of the output holds T_j(scale * x). 

            This rescales the input domain onto the standard Chebyshev interval 
            [-1, 1]: if the inputs x naturally live in [a, b], choosing an appropriate 
            scale (together with any centering done upstream) maps them into [-1, 1], 
            where the Chebyshev polynomials are well-conditioned and orthogonal. A 
            scale of 1.0 leaves the input unchanged and recovers the standard 
            Chebyshev polynomials T_j(x). 
        '''
        scale = self.scale
        x = x.flatten()
        out = torch.zeros(x.shape[0], self.degree + 1, dtype = x.dtype, device = x.device) 
        
        out[:, 0] = 1.0
        if self.degree >= 1:
            out[:, 1] = scale * x
        
        for j in range(2, self.degree + 1):
            out[:, j] = 2 * scale * x * out[:, j - 1] - out[:, j - 2]
        
        return out
    
    def __repr__(self) -> str:
        '''
        Prints a debug representation of a Chebyshev dictionary, reporting the degree. Typically to be called
        by writing x in the REPL where x is a ChebyshevDictionary object, or by print(x). 
        '''
        return f"ChebyshevDictionary(degree = {self.degree}, scale = {self.scale})"