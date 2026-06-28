import torch

def shuffle_and_split(x : torch.Tensor, y : torch.Tensor, test_size : float = 0.3, quadrature_weights : torch.Tensor = None, shuffle : bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    '''
    Shuffles x, y, and optional quadrature weights with a shared permutation and splits them into train/test sets.
    '''
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of samples")

    if shuffle:
        perm = torch.randperm(x.shape[0], device = x.device)
        x_perm = x[perm]
        y_perm = y[perm]
    else:
        x_perm = x
        y_perm = y

    W_perm = None
    if quadrature_weights is not None:
        if quadrature_weights.shape[0] != x.shape[0]:
            raise ValueError(f"Number of quadrature weights ({quadrature_weights.shape[0]}) is not equal to the number of samples ({x.shape[0]})")
        W_perm = quadrature_weights.to(device = x.device, dtype = x.dtype)
        if shuffle:
            W_perm = W_perm[perm]

    n_train = int((1 - test_size) * x.shape[0])
    x_train, x_test = x_perm[:n_train], x_perm[n_train:]
    y_train, y_test = y_perm[:n_train], y_perm[n_train:]
    W_train = None if W_perm is None else W_perm[:n_train]
    W_test = None if W_perm is None else W_perm[n_train:]

    return x_train, x_test, y_train, y_test, W_train, W_test

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

def complex_to_real_dtype(dtype : torch.dtype) -> torch.dtype:
    '''
    Helper function to convert a complex dtype to the corresponding real dtype. If the input is not a complex dtype, it is returned unchanged. 

    Parameters 
    ----------------------------------
    dtype : torch.dtype
        The dtype to convert to a real dtype if it is a complex dtype. 

    Returns 
    ----------------------------------
    torch.dtype
        The corresponding real dtype if the input was a complex dtype, otherwise the input dtype unchanged. 

    Raises 
    ----------------------------------
    None
    '''
    if dtype == torch.complex128:
        return torch.float64
    elif dtype == torch.complex64:
        return torch.float32
    else:
        return torch.float32 
    
def to_complex_dtype(tensor : torch.Tensor) -> torch.Tensor:
    '''
    Helper function to convert a tensor to a complex dtype. If the input is already a complex dtype, it is returned unchanged. 

    Parameters 
    ----------------------------------
    tensor : torch.Tensor
        The tensor to convert to a complex dtype. 

    Returns 
    ----------------------------------
    torch.Tensor
        The input tensor, converted to a complex dtype if it was not already a complex dtype. 

    Raises 
    ----------------------------------
    None
    '''
    if tensor.dtype == torch.float64:
        return tensor.to(torch.complex128)
    elif tensor.dtype == torch.float32:
        return tensor.to(torch.complex64)

def benchmark_metrics(dictionary: 'Dictionary', x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    '''
    Computes benchmark metrics for the given dictionary and trajectories.

    Parameters
    -------------------------
    dictionary : Dictionary
        The dictionary for which to compute the metrics.
    x : torch.Tensor
        Tensor of x values, shape (M, d).
    y : torch.Tensor
        Tensor of y values, shape (M, d).
    
    Returns
    -------------------------
    dict[str, torch.Tensor | dict[str, torch.Tensor]]
        A dictionary containing the computed metrics. The keys are:
        - 'Psi_X': The dictionary evaluated at x.
        - 'Psi_Y': The dictionary evaluated at y.
        - 'Lambda': The eigenvalues of the EDMD operator.
        - 'K': The EDMD matrix.
        - 'cond_num': The condition number of the weighted dictionary matrix.
        - 'loss': The loss computed from the singular values and residuals.
        - 'forecast_error': The forecast error of the EDMD operator.
        - 'pseudospec': The pseudospectra of the EDMD operator.
    '''
    from pyresdmd.compute.spectra import (
        EDMD,
        compute_eigendecomposition_from_weights,
        compute_forecast_error,
        compute_loss,
        compute_pseudospectra,
        compute_residuals,
    )

    Psi_X = dictionary.evaluate(x)
    Psi_Y = dictionary.evaluate(y)
    W = torch.ones(x.shape[0], device = x.device) / x.shape[0]
    Lambda, V = compute_eigendecomposition_from_weights(Psi_X, Psi_Y, W)
    K = EDMD(Psi_X, Psi_Y, W)
    residuals = compute_residuals(Lambda, V, Psi_X, Psi_Y, W)
    W_sqrt = torch.sqrt(W).unsqueeze(1)
    W_sqrtPsi_X = W_sqrt * Psi_X
    singvals = torch.linalg.svdvals(W_sqrtPsi_X)
    cond_num = (singvals[0]) / (singvals[-1])
    loss = compute_loss(singvals, residuals)
    forecast_error = compute_forecast_error(Psi_X, Psi_Y, K)
    pseudospec = compute_pseudospectra(Psi_X, Psi_Y)

    return {
        'Psi_X': Psi_X,
        'Psi_Y': Psi_Y,
        'Lambda': Lambda,
        'K': K,
        'cond_num': cond_num,
        'loss': loss,
        'forecast_error': forecast_error,
        'pseudospec': pseudospec,
    }