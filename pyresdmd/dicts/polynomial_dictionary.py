import torch
import torch.nn.functional as F
from pyresdmd.dicts.dictionary import Dictionary
from itertools import combinations_with_replacement

class PolynomialDictionary(Dictionary):
    def __init__(self, input_dim : int, degree : int) -> None:
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
            combos = list(combinations_with_replacement(range(input_dim), d))

            if d == 0:
                powers.append(torch.zeros((1, input_dim), dtype = torch.long))
                continue

            combo_tensor = torch.tensor(combos, dtype = torch.long)

            degree_powers = F.one_hot(combo_tensor, num_classes = input_dim).sum(dim = 1)
            # eg combo_tensor = [[1, 1, 2]], input_dim = 4 gives [[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
            # summing across columns gives [0, 2, 1, 0] which translates to x_1^0 * x_2^2 * x_3 * x_4^0 as expected
            powers.append(degree_powers)

        self.register_buffer('powers', torch.cat(powers, dim = 0))
        
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