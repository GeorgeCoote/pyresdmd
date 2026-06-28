import torch

def _quadrature_weights(M, quadrature_weights : torch.Tensor | None = None, device : torch.device | str | None = None, dtype : torch.dtype | None = None) -> torch.Tensor:
    '''
    Helper function which handles quadrature weights. If none are given, we default to uniform weights.
    '''
    from pyresdmd.utils.helpers import complex_to_real_dtype

    dtype = complex_to_real_dtype(dtype)

    if quadrature_weights is None:
        W = torch.ones(M, device = device, dtype = dtype) / M
    else:
        if quadrature_weights.ndim != 1 or quadrature_weights.shape[0] != M:
            raise ValueError(f"quadrature_weights must have shape ({M},)")
        if torch.is_complex(quadrature_weights):
            raise ValueError("quadrature_weights must be real-valued")
        if not torch.isfinite(quadrature_weights).all():
            raise ValueError("quadrature_weights must be finite")
        if (quadrature_weights < 0).any():
            raise ValueError("quadrature_weights must be non-negative")
        W = quadrature_weights.to(device = device, dtype = dtype)

    return W

def EDMD(Psi_X : torch.Tensor, Psi_Y : torch.Tensor, W : torch.Tensor, 
    ridge : float = 3e-1,
    factors : list[float] = None
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
        The lifted data matrix Psi_X 
    Psi_Y : torch.Tensor 
        The lifted data matrix Psi_Y 
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
    if factors is None:
        factors = [1.0, 10.0, 100.0, 1000.0] # default
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
    except RuntimeError:
        pass
    
    # double precision has failed
    
    # try a load of ridges 
    N = A.shape[1]
    ATB = (A.T @ B)
    ATA = A.T @ A
    
    for f in factors:
        coef = ridge * f
        ATA_reg = ATA + coef * torch.eye(N, device = A.device, dtype = A.dtype)
        
        try:
            K_candidate = torch.linalg.solve(ATA_reg, ATB)
            K_candidate = K_candidate.to(A.dtype)
            if torch.isfinite(K_candidate).all():
                return K_candidate 
        
        except RuntimeError:
            continue
    
    # fallback has failed
    raise ValueError("EDMD fail after exhausting regularizing factors. Perhaps try stronger regularization.")

def compute_eigendecomposition_from_weights(Psi_X : torch.Tensor, Psi_Y : torch.Tensor, W : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Strict eigendecomposition entry point that assumes quadrature weights are already resolved.
    '''
    K = EDMD(Psi_X, Psi_Y, W)
    return torch.linalg.eig(K)

def compute_eigendecomposition(Psi_X : torch.Tensor, Psi_Y : torch.Tensor, quadrature_weights : torch.Tensor = None, dtype : torch.dtype = None) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Convenience wrapper for eigendecomposition that resolves quadrature weights before solving.

    For stricter control, call compute_eigendecomposition_from_weights with a pre-validated weight tensor.
    
    Parameters 
    ----------------------------------
    Psi_X : torch.Tensor 
        The lifted data matrix Psi_X
    Psi_Y : torch.Tensor 
        The lifted data matrix Psi_Y 
    quadrature_weights : torch.Tensor 
        Torch tensor of quadrature weights
    same_device : bool
        If W is not on the same device as Psi_X, it will be moved there if this is set to True.
    dtype : torch.dtype
        Dtype used when constructing/coercing quadrature weights. If None, defaults to Psi_X.dtype
        (mapped to the corresponding real dtype for complex Psi_X).
    
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
    device = Psi_X.device
    if dtype is None:
        dtype = Psi_X.dtype
    W = _quadrature_weights(M, quadrature_weights, device, dtype)

    return compute_eigendecomposition_from_weights(Psi_X, Psi_Y, W)

def compute_residuals(Lambda : torch.Tensor, V : torch.Tensor, Psi_X : torch.Tensor, Psi_Y : torch.Tensor, W : torch.Tensor) -> torch.Tensor:
    '''
    Computes ResDMD residuals based off EDMD matrix, eigendecomposition, lifted data matrices and quadrature weights. 
    
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
        Lifted data matrix for x
    
    Psi_Y : torch.Tensor 
        Lifted data matrix for y 
    
    W : torch.Tensor 
        Matrix of quadrature weights
    
    Returns 
    ----------------------------------
    torch.Tensor 
        Tensor of residuals for each eigenpair 
    '''
    M = Psi_X.shape[0]

    if W.shape[0] != M:
        raise ValueError(f"Number of quadrature weights ({W.shape[0]}) is not equal to the number of snapshots ({Psi_X.shape[0]})")

    W_sqrt = torch.sqrt(W).unsqueeze(1)

    # need to cast to complex dtype because Lambda is complex
    complex_dtype = torch.complex128 if Psi_X.dtype == torch.float64 else torch.complex64
    W_sqrtPsi_X = (W_sqrt * Psi_X).to(complex_dtype)
    W_sqrtPsi_X_V = W_sqrtPsi_X @ V
    W_sqrtPsi_Y = (W_sqrt * Psi_Y).to(complex_dtype)
    
    diff = W_sqrtPsi_Y @ V - (W_sqrtPsi_X_V) * Lambda.unsqueeze(0) # = [... - ... * lambda_1, ... - ... * lambda_2, ...] etc.
    numerators = torch.linalg.vector_norm(diff, ord = 2, dim = 0)
    denominators = torch.linalg.vector_norm(W_sqrtPsi_X_V, ord = 2, dim = 0)

    if (denominators == 0).any():
        raise ValueError("One or more zero eigenvectors. This suggests a degenerate dictionary or quadrature weights.")
    
    return (numerators / denominators).real 

def compute_loss(singvals : torch.Tensor, residuals : torch.Tensor, 
    eps : float = 1e-8, loss_threshold : float = 1e3, use_cond_penalty : bool = True, penalty_coef : float = 1e-2,
) -> torch.Tensor:
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
        Singular values of the weighted Hankel matrix W_sqrt * Psi_X. Must be sorted in non-increasing order.
    residuals : torch.Tensor 
        ResDMD residuals computed by compute_residuals.
    eps : float 
        Offset in logarithm of condition number for numerical stability (for singular values near zero) 
    loss_threshold : float 
        Minimum condition number to penalize. 
    use_cond_penalty : bool
        Enable/disable condition penalty
    penalty_coef : float 
        Penalty assigned to condition number
    
    Returns 
    ----------------------------------
    Loss computed by the above formula.
    '''
    if singvals.ndim != 1 or singvals.numel() == 0:
        raise ValueError("singvals must be a non-empty 1D tensor")
    if residuals.ndim != 1 or residuals.shape[0] != singvals.shape[0]:
        raise ValueError("residuals must be a 1D tensor of the same length as singvals")
    if not torch.all(singvals[:-1] >= singvals[1:]):
        raise ValueError("singvals must be sorted in non-increasing order")

    N = residuals.shape[0]
    
    log_kappa = torch.log(singvals[0] + eps) - torch.log(singvals[-1] + eps)
    log_kappa_thresh = torch.log(torch.tensor(loss_threshold, device = singvals.device))
    
    cond_penalty = torch.relu(log_kappa - log_kappa_thresh) if use_cond_penalty else torch.zeros_like(log_kappa)
    
    return (1/N)*torch.sum(residuals * residuals) + penalty_coef * cond_penalty

def compute_forecast_error(Psi_X : torch.Tensor, Psi_Y : torch.Tensor, K : torch.Tensor) -> torch.Tensor:
    '''
    Computes the normalized one-step EDMD forecast error:
        sum_i ||psi(y^(i)) - K psi(x^(i))||_2^2 / (sum_i ||psi(y^(i))||_2^2)
    using all dictionary functions.
    '''
    if Psi_X.ndim != 2 or Psi_Y.ndim != 2:
        raise ValueError("Psi_X and Psi_Y must be 2D tensors")
    if Psi_X.shape != Psi_Y.shape:
        raise ValueError("Psi_X and Psi_Y must have the same shape")
    if K.ndim != 2:
        raise ValueError("K must be a 2D tensor")

    N = Psi_X.shape[1]
    if K.shape != (N, N):
        raise ValueError(f"K must have shape ({N}, {N})")

    if torch.isclose(torch.sum(torch.abs(Psi_Y)), torch.tensor(0.)):
        raise ValueError("Psi_Y is zero, so all included dictionary functions are zero at all evolved snapshots. This suggests an insufficient or faulty dictionary.")
    pred = Psi_X @ K
    diff = Psi_Y - pred
    numerator = torch.sum(torch.abs(diff) ** 2).real 
    denominator = torch.sum(torch.abs(Psi_Y) ** 2).real 

    return numerator/denominator

def compute_pseudospectra(Psi_X : torch.Tensor, Psi_Y : torch.Tensor, quadrature_weights : torch.Tensor | None = None,
                          re_range : tuple[float, float] = (-1.2, 1.2), im_range : tuple[float, float] = (-1.2, 1.2),
                          grid_resolution : int = 100, chunk_size : int = 512, save : bool = False, filename : str = 'pseudospectra.pt'
                          ) -> dict[str, torch.Tensor]:
    '''
    Computes tau(z) (an approximation to dist(z, Sp(K))) over a grid of points z in the complex plane, where K is the true Koopman operator. The formula for tau(z) is:
        tau(z) = min_{||v||=1} ||(W^(1/2) Psi_Y - z W^(1/2) Psi_X) v||_2
    where W is the diagonal matrix of quadrature weights.

    Parameters
    ----------------------------------
    Psi_X : torch.Tensor
        The lifted data matrix Psi_X, shape (M, N)
    Psi_Y : torch.Tensor
        The lifted data matrix Psi_Y, shape (M, N)
    quadrature_weights : torch.Tensor | None
        Optional quadrature weights, shape (M,). If None, defaults to uniform weights.
    re_range : tuple[float, float]
        Range of real parts to compute pseudospectrum over. Default (-1.2, 1.2).
    im_range : tuple[float, float]
        Range of imaginary parts to compute pseudospectrum over. Default (-1.2, 1.2).
    grid_resolution : int
        Number of points in each dimension to compute pseudospectrum over. Default 100.
    chunk_size : int
        Number of gridpoints to process in each batch when computing pseudospectrum. Default 512.
    
    '''
    device = Psi_X.device 
    M, N = Psi_X.shape 
    dtype = Psi_X.dtype

    W = _quadrature_weights(M, quadrature_weights, device, dtype)

    from pyresdmd.utils.helpers import force_h, force_h_vectorized

    W_sqrt = torch.sqrt(W).unsqueeze(1)

    W_sqrt_Psi_X = W_sqrt * Psi_X
    W_sqrt_Psi_Y = W_sqrt * Psi_Y

    # first, QR decompose W_sqrt_Psi_X 
    _, R = torch.linalg.qr(W_sqrt_Psi_X, mode = 'reduced')
    R = R.to(torch.complex128)

    R_inv = torch.linalg.inv(R)
    R_inv_conj_T = R_inv.conj().T 

    W_sqrt_Psi_X_complex = W_sqrt_Psi_X.to(torch.complex128)
    W_sqrt_Psi_Y_complex = W_sqrt_Psi_Y.to(torch.complex128)

    cross_corr = R_inv_conj_T @ (W_sqrt_Psi_X_complex.T.conj() @ W_sqrt_Psi_Y_complex) @ R_inv 
    gram_matrix = R_inv_conj_T @ (W_sqrt_Psi_Y_complex.T.conj() @ W_sqrt_Psi_Y_complex) @ R_inv

    cross_corr_conj_T = cross_corr.conj().T

    gram_matrix = force_h(gram_matrix)

    re_vals = torch.linspace(*re_range, grid_resolution, device = device)
    im_vals = torch.linspace(*im_range, grid_resolution, device = device)
    real_grid, imag_grid = torch.meshgrid(re_vals, im_vals, indexing='ij')
    Z_flat = torch.complex(real_grid.to(torch.float64), imag_grid.to(torch.float64)).reshape(-1)
    tau_flat = torch.zeros(len(Z_flat), dtype=torch.float32, device = device)

    identity = torch.eye(N, dtype = torch.complex128, device = device)

    for start in range(0, len(Z_flat), chunk_size):
        end = min(start + chunk_size, len(Z_flat))
        Z_batch = Z_flat[start:end].to(device)
        Z_batch_conj = Z_batch.conj()

        Z_vec = Z_batch[:, None, None]
        Z_conj_vec = Z_batch_conj[:, None, None]

        A_batch = (
            gram_matrix[None]
            - Z_vec * cross_corr_conj_T[None]
            - Z_conj_vec * cross_corr[None]
            + (Z_conj_vec * Z_vec) * identity[None]
        )

        A_batch = force_h_vectorized(A_batch)

        eigvals = torch.linalg.eigvalsh(A_batch)

        tau_flat[start:end] = (
            eigvals[:, 0].real.clamp(min = 0).sqrt().to(dtype = torch.float32, device = device)
        )
    
    if save:
        torch.save(tau_flat.reshape(grid_resolution, grid_resolution).cpu(), filename)
    
    return {
        're_vals': re_vals.cpu(),
        'im_vals': im_vals.cpu(),
        'tau_grid': tau_flat.reshape(grid_resolution, grid_resolution).cpu()
    }
