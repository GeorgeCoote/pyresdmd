import pytest 
import torch 

from pyresdmd.compute.spectra import _quadrature_weights, compute_loss, compute_forecast_error

# test _quadrature_weights

def test_quadrature_weights_None() -> None:
    '''Tests quadrature weight initialization with no specified weights'''
    M = 100
    expected = 0.01 * torch.ones(M)
    W = _quadrature_weights(M, quadrature_weights = None)
    
    assert torch.allclose(W, expected)
  
def test_quadrature_weights_specified() -> None:
    '''Tests quadrature weight initialization with specified weights'''
    M = 100
    quadrature_weights = torch.arange(100)
    W = _quadrature_weights(M, quadrature_weights = quadrature_weights)
    
    assert torch.allclose(W, quadrature_weights.to(dtype = torch.float32)) 

def test_quadrature_weights_wrong_ndim() -> None:
    '''Tests quadrature weight with ndim > 1'''
    M = 50
    quadrature_weights = torch.arange(M).reshape((25, 2))
    
    with pytest.raises(ValueError):
        _quadrature_weights(M, quadrature_weights = quadrature_weights)

def test_quadrature_weights_wrong_shape() -> None:
    '''Tests quadrature weight shape check'''
    M = 50
    quadrature_weights = torch.arange(25)
    
    with pytest.raises(ValueError):
        _quadrature_weights(M, quadrature_weights = quadrature_weights)

def test_quadrature_weights_complex() -> None:
    '''Tests whether _quadrature_weights rejects complex weights.'''
    M = 50
    quadrature_weights = 1j*torch.arange(50)
    
    with pytest.raises(ValueError):
        _quadrature_weights(M, quadrature_weights = quadrature_weights)

def test_quadrature_weights_infinite() -> None:
    '''Tests whether _quadrature_weights rejects infinite weights.'''
    M = 3
    quadrature_weights = torch.tensor([1., torch.inf, torch.pi])
    
    with pytest.raises(ValueError):
        _quadrature_weights(M, quadrature_weights = quadrature_weights)

def test_quadrature_weights_negative() -> None:
    '''Tests whether _quadrature_weights rejects negative weights.'''
    M = 5
    quadrature_weights = torch.tensor([1., 0., -1., -2., torch.pi])
    
    with pytest.raises(ValueError):
        _quadrature_weights(M, quadrature_weights = quadrature_weights)

def test_zero_quadrature_weights() -> None:
    '''Tests whether _quadrature_weights rejects all zero weights.'''
    M = 4
    quadrature_weights = torch.zeros(M)
    
    _quadrature_weights(M, quadrature_weights = quadrature_weights)

# test compute_loss

def test_compute_loss_zeros() -> None:
    '''Tests compute_loss with an all-zero example'''
    N = 50
    singvals = torch.ones(N)
    residuals = torch.zeros(N)
    expected = torch.tensor(0.)
    loss = compute_loss(singvals, residuals, eps = 0.0)
    assert torch.allclose(loss, expected)

def test_compute_loss_non_trivial() -> None:
    '''Test compute_loss with non-trivial singvals and residuals.'''
    singvals = torch.tensor([0.9821, 0.9601, 0.9381, 0.8146, 0.7971, 0.7450, 0.6895, 0.6443, 0.6113,
        0.5586, 0.4255, 0.4239, 0.3587, 0.2580, 0.2283, 0.1212, 0.0375, 0.0343,
        0.0228, 0.0068]) # log(0.9821) - log(0.068) = 4.9728. just random numbers I generated ~ U[0, 1]
    residuals = torch.tensor([0.0899, 0.1897, 0.0615, 0.0794, 0.1595, 0.0499, 0.2318, 0.2981, 0.1444,
        0.2863, 0.2999, 0.2014, 0.0175, 0.1516, 0.0638, 0.2439, 0.1479, 0.2153,
        0.1834, 0.0892]) # sum of squares = 0.0327, ~ U[0, 0.3]
    loss_threshold = torch.tensor(2.) 
    # log_kappa - log_kappa_threshold = 4.2796
    # 4.2796/100 + 0.0327 = 0.0755 (+error)
    loss = compute_loss(singvals, residuals, eps = 0., loss_threshold = 2.)
    assert torch.allclose(loss, torch.tensor(0.0755 + 2.9638e-05))

def test_compute_loss_empty_singvals() -> None:
    '''Tests that compute_loss raises ValueError when singvals is empty.'''
    singvals = torch.tensor([])
    residuals = torch.tensor([0.1, 0.2, 0.3])
    
    with pytest.raises(ValueError):
        compute_loss(singvals, residuals)

def test_compute_loss_singvals_not_sorted() -> None:
    '''Tests that compute_loss raises ValueError when singvals is not sorted in non-increasing order.'''
    singvals = torch.tensor([1., 0.5, 0.8]) 
    residuals = torch.tensor([0.1, 0.2, 0.3])
    
    with pytest.raises(ValueError):
        compute_loss(singvals, residuals)

def test_compute_loss_wrong_ndim() -> None:
    '''Tests that compute_loss raises ValueError when singvals or residuals have ndim != 1.'''
    singvals = torch.tensor([[1., 0.5], [0.8, 0.3]])
    residuals = torch.tensor([0.1, 0.2, 0.3])
    
    with pytest.raises(ValueError):
        compute_loss(singvals, residuals)

    singvals = torch.tensor([1., 0.5, 0.8])
    residuals = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    
    with pytest.raises(ValueError):
        compute_loss(singvals, residuals)

def test_compute_loss_residuals_wrong_ndim() -> None:
    '''Tests that compute_loss raises ValueError when residuals has ndim != 1.'''
    singvals = torch.tensor([1., 0.5, 0.8])
    residuals = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    
    with pytest.raises(ValueError):
        compute_loss(singvals, residuals)

def test_compute_loss_residuals_wrong_length() -> None:
    '''Tests that compute_loss raises ValueError when residuals has different length than singvals.'''
    singvals = torch.tensor([1., 0.5, 0.8])
    residuals = torch.tensor([0.1, 0.2]) # length 2 instead of 3
    
    with pytest.raises(ValueError):
        compute_loss(singvals, residuals)

# compute_forecast_error

def test_compute_forecast_error_non_trivial() -> None:
    '''Test compute_forecast_error with non-trivial Psi_X, Psi_Y, K.'''
    x = 2*torch.pi*torch.tensor(1/5)
    K = torch.tensor([[torch.cos(x), -torch.sin(x)], [torch.sin(x), torch.cos(x)]])
    Psi_X = torch.tensor([[1., 0.], [torch.cos(x), torch.sin(x)], [torch.cos(2*x), torch.sin(2*x)]])
    Psi_Y = torch.tensor([[0.31, -0.95], [1., 0.], [0.31, 0.95]])
    # numerator = 4.1654e-06
    # denominator = 2.9972
    # forecast_error = 1.3897e-06
    forecast_error = compute_forecast_error(Psi_X, Psi_Y, K)
    assert torch.allclose(forecast_error, torch.tensor(1.3897e-06))

def test_compute_forecast_error_zero_Psi_Y() -> None:
    '''Tests compute_forecast_error with zero Psi_Y, which should yield zero forecast error.'''
    Psi_X = torch.tensor([[1., 0.], [0., 1.]])
    Psi_Y = torch.zeros_like(Psi_X)
    K = torch.eye(2)
    with pytest.raises(ValueError):
        compute_forecast_error(Psi_X, Psi_Y, K)

def test_compute_forecast_error_wrong_ndim() -> None:
    '''Tests compute_forecast_error shape with misshaped Psi_X, Psi_Y and K.'''
    Psi_X = torch.tensor([1., 0., 2.])
    Psi_Y = torch.tensor([[1., 0.], [0., 1.]])
    K = torch.eye(2)

    with pytest.raises(ValueError):
        compute_forecast_error(Psi_X, Psi_Y, K)

    Psi_X = torch.tensor([[1., 0.], [0., 1.]])
    Psi_Y = torch.tensor([[1., 0.], [0., 1.]])
    K = torch.tensor([1., 0.])

    with pytest.raises(ValueError):
        compute_forecast_error(Psi_X, Psi_Y, K)

def test_compute_forecast_error_shape_mismatch() -> None:
    '''Tests compute_forecast_error shape checks for incompatible matrix shapes.'''
    Psi_X = torch.tensor([[1., 0.], [0., 1.]])
    Psi_Y = torch.tensor([[1., 0., 2.], [0., 1., 3.]])
    K = torch.eye(2)

    with pytest.raises(ValueError):
        compute_forecast_error(Psi_X, Psi_Y, K)

    Psi_X = torch.tensor([[1., 0.], [0., 1.]])
    Psi_Y = torch.tensor([[1., 0.], [0., 1.]])
    K = torch.eye(3)

    with pytest.raises(ValueError):
        compute_forecast_error(Psi_X, Psi_Y, K)
