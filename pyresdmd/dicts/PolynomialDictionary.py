from pyresdmd.dicts import BaseDictionary
import torch
from itertools import combinations_with_replacement, product

class PolynomialDictionary(BaseDictionary):
    def __init__(self, input_dim : int, degree : int):
        '''
        Initializes polynomial dictionary. 
        
        Parameters
        -------------------------
        input_dim : int 
            Input dimension for dictionary. 
        
            For example, input dimension of 1 supports polynomials such as x, x^4 + 1, while dimension 3 would 
        support xy + z^3
        
        degree : int 
            Degree of polynomials in dictionary. 
            
            For example, x^2 + 1 has degree 1 and xy + z^3 has degree 3. 
        
        Returns
        -------------------------
        None, just sets class attributes.
        '''
        
        super().__init__()
        self._input_dim = input_dim 
        self._degree = degree 
        
        powers = []
        for d in range(degree + 1):
            for combo in combinations_with_replacement(range(input_dim), d):
                # this iterates over d-tuples with elements <= input_dim
                # the number of appearances of k corresponds to the largest power of x_k that should appear in the polynomial.
                # for example (1, 1, 2, 3) means that we allow squared terms in x and linear terms in y, z.
                
                p = [0] * input_dim 
                for idx in combo:
                    p[idx] += 1
                powers.append(p)
           
        self.register_buffer('powers', torch.tensor(powers))
        
    @property 
    def size(self) -> int:
        '''
        Returns the size of the dictionary. 
        '''
        return self.powers.shape[0]
     
    @property 
    def input_dim(self) -> int:
        '''
        Returns the shared input dimension of the dictionary functions.
        '''
        return self._input_dim 
    
    def evaluate(self, x : torch.Tensor) -> torch.Tensor:
        '''
        Evaluates the dictionary at the point x. 
        
        Parameters 
        -------------------------
        x : torch.Tensor 
            The point at which to evaluate. 
        
        Returns 
        -------------------------
        torch.Tensor 
            A tensor which represents every dictionary function evaluated at x.
        '''
        return torch.prod(
            x.unsqueeze(1) # changes shape from (M, d) to (M, 1, d)
            ** 
            self.powers.unsqueeze(0), # changes shape from (N, d) to (1, N, d)
            # at this point we have a matrix whose (m, n, k) entry is equal to x[m, k]**powers[n, k]
            dim = 2 # multiplies across the last dimension, so the (m, n)th entry is x[m, 0] ** powers[n, 0] * x[m, 1] ** powers[n, 1] * ... * x[m, d - 1] ** powers[n, d - 1], precisely what we want
        )
    
    def __repr__(self) -> str:
        '''
        Prints a debug representation of a polynomial dictionary, recording the input dimension 
        of the dictionary as well as the maximum degree. Typically called by writing x in the REPL
        where x is a PolynomialDictionary object, or by print(x). 
        '''
        return f"PolynomialDictionary(input_dim = {self._input_dim}, degree = {self._degree})"
