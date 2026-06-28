import torch
from collections.abc import Callable

def hankel_matrices(g: Callable[[torch.Tensor], torch.Tensor], x : torch.Tensor, M : int, N : int) -> tuple[torch.Tensor, torch.Tensor]: #gh
    '''
    Produces Hankel matrices associated with an observable g. 
    
    Given a trajectory x_0, x_1, ..., x_(M + N - 1) and an observable g, we create the matrices 
        Psi_X[i, j] = g(x_(i + j)) where 0 <= i <= M - 1, 0 <= j <= N - 1
        Psi_Y[i, j] = g(x_(i + j + 1)) where 0 <= i <= M - 1, 0 <= j <= N - 1
    
    g should support vectorized calculations with tensors. 
    
    Parameters
    ----------------------------------
    g
        Function supporting pytorch tensors and vectorization. 
        
        E.g. may be a function involving torch.sin, torch.exp etc.
        
    x : torch.Tensor 
        Trajectory data. 
        
    M : torch.Tensor 
        Dictionary size. 
        
    N : torch.Tensor 
        Trajectory length for each starting point.
    
    Returns 
    ----------------------------------
    tuple[torch.Tensor, torch.Tensor]
        Two tensors representing the Hankel matrices Psi_X and Psi_Y. 
    '''
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x)}")
    if not isinstance(M, int) or not isinstance(N, int):
        raise TypeError(f"M and N must be integers, got {type(M)} and {type(N)}")
    if (N <= 0) or (M <= 0):
        raise ValueError(f"M and N must be positive integers, got M = {M} and N = {N}")
    if x.shape[0] != M + N:
        raise ValueError(f"x has incorrect size. Expected {M + N}, got {x.shape[0]}")
    if x.shape != g(x).shape:
        raise ValueError(f"g produces output of incorrect shape. Expected {x.shape}, got {g(x).shape}")

    gx = g(x)
    rows = torch.arange(M, device = x.device).unsqueeze(1) # change shape from (M,) to (M, 1)
    cols = torch.arange(N, device = x.device).unsqueeze(0) # change shape from (M,) to (1, M)
    idx = rows + cols # creates idx[i, j] = i + j
    
    return gx[idx], gx[idx + 1]
