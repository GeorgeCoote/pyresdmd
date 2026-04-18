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

def force_h(A : torch.Tensor) -> torch.Tensor:
    '''
    Helper function to eliminate spurious imaginary parts (arising due to floating-point/other numerical error) in a matrix that is theoretically required to be Hermitian.

    This is done by taking 1/2 (A + A^\ast)/2. This is always Hermitian. 

    Further, if A is symmetric, the thus produced matrix retains the same real parts with spurious imaginary parts eliminated.
    
    Failing to do this can result in Cholesky failing. 
    
    Parameters 
    ----------------------------------
    A : torch.Tensor
        Matrix to Hermitian-ize represented as tensor. 
    
    Returns 
    ----------------------------------
    torch.Tensor 
        Hermitian-ized vector represented as tensor. 
    
    Raises 
    ----------------------------------
    None
    '''
    if A.shape[0] != A.shape[1]:
        raise ValueError("A should be a square matrix.") 
    return 0.5 * (A + A.conj().T)

def force_h_vectorized(A : torch.Tensor) -> torch.Tensor:
    '''
    A vectorized version of force_h. Input should be a tensor of matrices (that is, a 3D tensor), each of which need to be Hermitianized.
    
    This is significantly faster than looping over all such matrices in Python and Hermitianizing them individually. 
    
    Parameters 
    ----------------------------------
    A : torch.Tensor
        3D tensor of matrices to Hermitianize in the sense introduced in force_h. 
    
    Returns 
    ----------------------------------
    torch.Tensor 
        3D tensor of Hermitianized matrices. 
    
    Raises 
    ----------------------------------
    ValueError 
        If the size of the second and third dimension of the tensor are unequal, then the input tensor does not consist of 
        square matrices, and hence trying to Hermitianize them will either result in unexpected behaviour or dimension 
        errors.
    '''
    if A.shape[1] != A.shape[2]:
        raise ValueError("Tensor A should consist of square matrices")
    return 0.5 * (A + A.conj().transpose(-1, -2)) # transpose last two dimensions
