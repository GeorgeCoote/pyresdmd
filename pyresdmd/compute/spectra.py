import torch

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
        raise ValueError("Number of quadrature weights ({W.shape[0]}) is not equal to the number of snapshots ({Psi_X.shape[0]})")
    
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
