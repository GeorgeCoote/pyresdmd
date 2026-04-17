import torch

def dist2(x : torch.Tensor, centres : torch.Tensor) -> torch.Tensor:
    '''
    Given trajectory data (x_i) and centres (y_j), this function produces a matrix A[i, j] = ||x_i - y_j||_2 
    
    Parameters
    -------------------------
    x : torch.Tensor 
        Trajectory data provided as a tensor of shape (trajectory_size, input_dim) 
    centres : torch.Tensor 
        Tensor of centres, shape (dictionary_size, input_dim) 
    
    Returns 
    -------------------------
    torch.Tensor
        The matrix A[i, j] = ||x_i - y_j||_2.
    
    Raises 
    -------------------------
    ValueError
        If x and centres differ on the second dimension.
    '''
    if x.shape[1] != centres.shape[1]:
        raise ValueError("x and centres should correspond to the same input dimension")
    # x has shape (trajectory_size, input_dim) and centres has shape (dictionary_size, input_dim)
    # abbreviate this (n, d) and (m, d)
    x = x.unsqueeze(1) # change shape to (n, 1, d)
    centres = centres.unsqueeze(0) # change shape to (1, m, d)
    diff = x - centres # broadcasts to shape (n, m, d), where diff[i, j, k] = x[i, k] - centres[j, k]
    return torch.linalg.vector_norm(diff, ord = 2, dim = 2) # this will sum out the last dimension
