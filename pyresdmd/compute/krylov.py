import torch

def krylov_matrices(g, x : torch.Tensor, M : int, N : int) -> tuple[torch.tensor, torch.tensor]: #gh
    '''
    Produces Hankel matrices associated with delay embedding. 
    
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
    if x.shape[0] != M + N:
        raise ValueError(f"x has incorrect size. Expected {M + N}, got {x.shape[0]}")
    gx = g(x)
    rows = torch.arange(M, device = gx.device).unsqueeze(1) # change shape from (M,) to (M, 1)
    cols = torch.arange(N, device = gx.device).unsqueeze(0) # change shape from (M,) to (1, M)
    idx = rows + cols # creates idx[i, j] = i + j
    
    return gx[idx], gx[idx + 1]
