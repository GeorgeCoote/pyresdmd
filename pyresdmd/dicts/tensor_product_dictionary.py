from pyresdmd.dicts.dictionary import Dictionary
import torch
from itertools import product


class TensorProductDictionary(Dictionary):
    def __init__(self, *dicts: Dictionary) -> None:
        '''
        Initializes tensor product dictionary from multiple dictionaries.
        
        Parameters
        -------------------------
        *dicts : Dictionary
            Arbitrary number of Dictionary objects to form the tensor product.
            The tensor product will be psi^(1)_(n_1)(x_1) * ... * psi^(d)_(n_d)(x_d),
            where psi^(1), ..., psi^(d) are the input dictionaries and x_1, ..., x_d
            are the corresponding input variables.
        
        Returns 
        -------------------------
        None, just sets class attributes.
        '''
        super().__init__()
        
        if len(dicts) == 0:
            raise ValueError("TensorProductDictionary requires at least one dictionary")
        
        self.dicts = dicts
        self._input_dim = sum(d.input_dim for d in dicts)
        self._size = 1
        for d in dicts:
            self._size *= d.size
    
    @property
    def size(self) -> int:
        '''
        Returns the size of the dictionary (product of all dictionary sizes).
        '''
        return self._size
    
    @property
    def input_dim(self) -> int:
        '''
        Returns the input dimension (sum of all dictionary input dimensions).
        '''
        return self._input_dim
    
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Evaluates the tensor product dictionary at the point x.
        
        Parameters
        -------------------------
        x : torch.Tensor
            The point(s) at which to evaluate, shape (M, d) where M is the number
            of points and d is the total input dimension (sum of all dictionary
            input dimensions).
        
        Returns
        -------------------------
        torch.Tensor
            A tensor of shape (M, N) where N is the product of all dictionary sizes.
            Each entry (m, n) represents the tensor product basis function evaluated at x[m].
        '''
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Input x must have shape ({x.shape[0]}, {self.input_dim}), but got {x.shape}")
        # Split x into chunks for each dictionary
        # We split the shape x, (M, d), into (M, d_1), (M, d_2), ..., (M, d_k) where d_i is the input dimension of the ith dictionary
        x_chunks = []
        start_idx = 0
        for d in self.dicts:
            end_idx = start_idx + d.input_dim
            x_chunks.append(x[:, start_idx:end_idx])
            start_idx = end_idx
        
        # Evaluate each dictionary
        evals = [d.evaluate(x_chunk) for d, x_chunk in zip(self.dicts, x_chunks)]
        
        # Compute tensor product
        # Start with the first evaluation
        result = evals[0]  # shape (M, N_1)
        
        for eval_next in evals[1:]:
            # Result shape (M, N_current), eval_next shape (M, N_next) -> (M, N_current * N_next)
            result = torch.einsum('mi,mj->mij', result, eval_next).reshape(result.shape[0], -1)
        
        return result
    
    def __repr__(self) -> str:
        '''
        Prints a debug representation of a tensor product dictionary.
        '''
        dict_strs = ", ".join(repr(d) for d in self.dicts)
        return f"TensorProductDictionary({dict_strs})"