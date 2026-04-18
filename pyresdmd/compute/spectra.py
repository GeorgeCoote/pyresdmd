import torch

def _quadrature_weights(M, quadrature_weights = None, device = None) -> torch.Tensor:
    '''
    Helper function which handles quadrature weights. If none are given, we default to uniform weights.
    
    If weights are provided, we check their size and move them to the appropriate device. 
    '''
    if quadrature_weights is not None:
        if quadrature_weights.shape[0] != M:
            raise ValueError(f"Number of quadrature weights ({quadrature_weights.shape[0]}) is not equal to the number of snapshots ({Psi_X.shape[0]})")
        
        W = quadrature_weights
        
        if device is not None:
            W = W.to(device) 
    
    else:
        W = torch.ones(M) / M
        
        if device is not None:
            W = W.to(device)
    
    return W

def EDMD(Psi_X : torch.Tensor, Psi_Y : torch.Tensor, W : torch.Tensor, 
    ridge : float = 3e-1,
    factors : list[float] = [1.0, 10.0, 100.0, 1000.0]
) -> torch.Tensor:
    '''
    Compute the EDMD matrix (W^(1/2) Psi_X)^\dagger W^(1/2) Psi_Y, where \dagger denotes the Moore-Penrose pseudoinverse 
        (https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse)
    
    By mathematical theory, this is the same as solving the least-squares optimization problem:
        ||(W^(1/2) Psi_X) A - W^(1/2) Psi_Y||_2
    for A. 
    
    We first try using torch's lstsq to find a solution. Failing this, we add a small regularizing term, trying increasingly larger regularizers before failing. 
    
    Parameters
    ----------------------------------
    Psi_X : torch.Tensor 
        The Hankel matrix Psi_X 
    Psi_Y : torch.Tensor 
        The Hankel matrix Psi_Y 
    W : torch.Tensor 
        Torch tensor of quadrature weights. 
    ridge : float 
        Default 3e-1 
        Ridge coefficient to use. 
    factors : list[float]
        Regularizing factors to use. Large factors risk finding a stable but low-accuracy solution, while small factors may not remove instability. 
    
    Returns 
    ----------------------------------
    torch.Tensor 
        EDMD matrix.
    
    Raises
    ----------------------------------
    ValueError 
        If number of quadrature weights is not equal to the number of snapshots. 
        If a quadrature weight is negative. 
        If Psi_X and Psi_Y have different shapes.
    '''
    if W.shape[0] != Psi_X.shape[0]:
        raise ValueError(f"Number of quadrature weights ({W.shape[0]}) is not equal to the number of snapshots ({Psi_X.shape[0]})")
    
    if (W < 0).any():
        raise ValueError("Quadrature weights must be non-negative.")
    
    if Psi_X.shape != Psi_Y.shape:
        raise ValueError("Psi_X and Psi_Y must have the same shape.")
    
    W_sqrt = torch.sqrt(W).unsqueeze(1) 
    
    W_sqrt_Psi_X = W_sqrt * Psi_X 
    W_sqrt_Psi_Y = W_sqrt * Psi_Y 
    
    # for brevity 
    A = W_sqrt_Psi_X 
    B = W_sqrt_Psi_Y
    
    # try bald least-squares first 
    try:
        sol = torch.linalg.lstsq(A, B).solution 
        if torch.isfinite(sol).all():
            return sol
    except Exception:
        pass 
    
    # at this point, bald least squares has failed to produce a solution
    
    # retry in double precision 
    A_solve = A.double()
    B_solve = B.double()
    
    try:
        sol = torch.linalg.lstsq(A_solve, B_solve).solution 
        K = sol.to(A.dtype)
        if torch.isfinite(K).all():
            return K
    except Exception:
        pass
    
    # double precision has failed
    
    # try a load of ridges 
    N = A.shape[1]
    ATB = (A.T @ B)
    
    for f in factors:
        coef = float(ridge) * f
        ATA = A.T @ A + coef * torch.eye(N, device = A.device, dtype = A.dtype)
        
        try:
            K_candidate = torch.linalg.solve(ATA, ATB)
            K_candidate = K_candidate.to(A.dtype)
            if torch.isfinite(K_candidate).all():
                return K_candidate 
        
        except Exception:
            continue
    
    # fallback has failed
    raise ValueError("EDMD fail after exhausting regularizing factors. Perhaps try stronger regularization.")

def compute_eigendecomposition(Psi_X : torch.Tensor, Psi_Y : torch.Tensor, quadrature_weights : torch.Tensor = None, same_device : bool = True) -> torch.Tensor:
    '''
    Computes the eigendecomposition of the EDMD matrix based off the Hankel matrices Psi_X and Psi_Y and quadrature weights. 
    
    Parameters 
    ----------------------------------
    Psi_X : torch.Tensor 
        The Hankel matrix Psi_X
    Psi_Y : torch.Tensor 
        The Hankel matrix Psi_Y 
    quadrature_weights : torch.Tensor 
        Torch tensor of quadrature weights
    same_device : bool
        If W is not on the same device as Psi_X, it will be moved there if this is set to True.
    
    Returns 
    ----------------------------------
    2-tuple
        (Lambda, V)
            The eigendecomposition of K 
    
    Raises 
    ----------------------------------
    ValueError
        If number of quadrature weights is not equal to the number of snapshots. 
    '''

    M = Psi_X.shape[0]
    device = Psi_X.device if same_device else None
    W = _quadrature_weights(M, quadrature_weights, device)

    K = EDMD(Psi_X, Psi_Y, W)

    return torch.linalg.eig(K)

def compute_residuals(Lambda : torch.Tensor, V : torch.Tensor, Psi_X : torch.Tensor, Psi_Y : torch.Tensor, W : torch.Tensor,
                     same_device : bool = True) -> torch.Tensor:
    '''
    Computes ResDMD residuals based off EDMD matrix, eigendecomposition, Hankel matrices and quadrature weights. 
    
    The formula for the ResDMD residual is:
        res(lambda_j, v_j) = ||(W^(1/2) Psi_Y - lambda_j W^{1/2} Psi_X) v_j||_2 / ||W^(1/2) Psi_X v_j||_2
    As usual this is computed in a vectorized way. 
    
    Of course since this is the quotient of two real numbers, it ought to be real. Since the vectors of concern may have 
    large imaginary parts that formally cancel, there is always a risk that the quantity has a spurious imaginary part. 
    
    Hence we technically compute:
        Re(||(W^(1/2) Psi_Y - lambda_j W^{1/2} Psi_X) v_j||_2 / ||W^(1/2) Psi_X v_j||_2)
    which mathematically is the same quantity.
    
    Parameters
    ----------------------------------
    Lambda : torch.Tensor 
        Eigenvalues in eigendecomposition
    
    V : torch.Tensor 
        Eigenvectors in eigendecomposition 
    
    Psi_X : torch.Tensor 
        Hankel matrix for x
    
    Psi_Y : torch.Tensor 
        Hankel matrix for y 
    
    W : torch.Tensor 
        Matrix of quadrature weights

    same_device : bool
        Moves quadrature weights to the same device as Psi_X, if it not already there.
    
    Returns 
    ----------------------------------
    torch.Tensor 
        Tensor of residuals for each eigenpair 
    '''
    M = Psi_X.shape[0]
    device = Psi_X.device if same_device else None

    W = _quadrature_weights(M, W, device)
    W_sqrt = torch.sqrt(W).unsqueeze(1)

    # need to cast to complex dtype because Lambda is complex
    complex_dtype = torch.complex128 if Psi_X.dtype == torch.float64 else torch.complex64
    W_sqrtPsi_X = (W_sqrt * Psi_X).to(complex_dtype)
    W_sqrtPsi_Y = (W_sqrt * Psi_Y).to(complex_dtype)
    
    diff = W_sqrtPsi_Y @ V - (W_sqrtPsi_X @ V) * Lambda.unsqueeze(0) # = [... - ... * lambda_1, ... - ... * lambda_2, ...] etc.
    numerators = torch.linalg.vector_norm(diff, ord = 2, dim = 0)
    denominators = torch.linalg.vector_norm(W_sqrtPsi_X @ V, ord = 2, dim = 0)
    
    return (numerators / denominators).real 

def compute_loss(singvals : torch.Tensor, residuals : torch.Tensor, 
    eps : float = 1e-8, loss_threshold : float = 1e3, condition_penalty : bool = True, penalty_coef : float = 1e-2,
) -> float:
    '''
    Computes loss function based off singular values of weighted Hankel matrix. 
    
    We use ResDMD residuals as well as a penalty term for large condition numbers. 

    The formula will be:
        loss((lambda_j, v_j)) = (1/N) sum_j |res(lambda_j, v_j)|^2 + c * ReLU(log(kappa) - log(kappa_0))
    where:
        N is the number of eigenpairs 
        (lambda_j, v_j) are the eigenpairs, and the sum goes over them 
        kappa is the condition number of the Hankel matrix Psi_X 
        kappa_0 is a large condition number past which we kick in a penalty. 
    Note that this loss is convex and differentiable away from kappa = kappa_0.
    
    Parameters 
    ----------------------------------
    singvals : torch.Tensor 
        Singular values of the weighted Hankel matrix W_sqrt * Psi_X 
    residuals : torch.Tensor 
        ResDMD residuals computed by compute_residuals.
    eps : float 
        Offset in logarithm of condition number for numerical stability (for singular values near zero) 
    loss_threshold : float 
        Minimum condition number to penalize. 
    penalty_coef : float 
        Penalty assigned to condition number
    
    Returns 
    ----------------------------------
    Loss computed by the above formula.
    '''
    condition_penalty_flag = condition_penalty # to avoid confusion with cond_penalty
    N = residuals.shape[0]
    
    log_kappa = torch.log(singvals[0] + eps) - torch.log(singvals[-1] + eps)
    log_kappa_thresh = torch.log(torch.tensor(loss_threshold, device = singvals.device))
    
    cond_penalty = torch.relu(log_kappa - log_kappa_thresh) if condition_penalty_flag else torch.zeros_like(log_kappa)
    
    return (1/N)*torch.sum(residuals * residuals) + penalty_coef * cond_penalty
