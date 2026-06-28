import pytest
import torch 

from pyresdmd.utils.helpers import shuffle_and_split, dist2, force_h, force_h_vectorized 

def test_shuffle_and_split_preserves_pairing_and_partition() -> None:
    '''Tests that shuffle_and_split returns a valid train/test partition while preserving x-y row pairing.'''
    x = torch.arange(20, dtype = torch.float32).reshape(10, 2)
    y = x.sum(dim = 1, keepdim = True)

    x_train, x_test, y_train, y_test, W_train, W_test = shuffle_and_split(x, y, test_size = 0.3)

    assert W_train is None
    assert W_test is None

    n_train_expected = int((1 - 0.3) * x.shape[0])
    assert x_train.shape[0] == n_train_expected
    assert x_test.shape[0] == x.shape[0] - n_train_expected
    assert y_train.shape[0] == n_train_expected
    assert y_test.shape[0] == x.shape[0] - n_train_expected

    x_concat = torch.cat([x_train, x_test], dim = 0)
    y_concat = torch.cat([y_train, y_test], dim = 0)

    # every shuffled row in x should retain its matching y row from the original tensors.
    for i in range(x_concat.shape[0]):
        mask = (x == x_concat[i]).all(dim = 1)
        idx = torch.where(mask)[0]
        assert idx.numel() == 1
        assert torch.allclose(y_concat[i], y[idx.item()])

def test_shuffle_and_split_with_quadrature_weights() -> None:
    '''Tests that quadrature weights are shuffled/split consistently and cast to x dtype/device.'''
    x = torch.arange(16, dtype = torch.float32).reshape(8, 2)
    y = 2 * x[:, :1]
    W = torch.arange(8, dtype = torch.float64)

    x_train, x_test, y_train, y_test, W_train, W_test = shuffle_and_split(x, y, test_size = 0.25, quadrature_weights = W)

    assert W_train is not None
    assert W_test is not None
    assert W_train.dtype == x.dtype
    assert W_test.dtype == x.dtype

    x_concat = torch.cat([x_train, x_test], dim = 0)
    y_concat = torch.cat([y_train, y_test], dim = 0)
    W_concat = torch.cat([W_train, W_test], dim = 0)

    for i in range(x_concat.shape[0]):
        mask = (x == x_concat[i]).all(dim = 1)
        idx = torch.where(mask)[0]
        assert idx.numel() == 1
        j = idx.item()
        assert torch.allclose(y_concat[i], y[j])
        assert torch.isclose(W_concat[i], W[j].to(dtype = x.dtype))

def test_shuffle_and_split_sequential_partition() -> None:
    '''Tests that shuffle_and_split can preserve sequential ordering when shuffle is disabled.'''
    x = torch.arange(20, dtype = torch.float32).reshape(10, 2)
    y = x.sum(dim = 1, keepdim = True)
    W = torch.arange(10, dtype = torch.float32)

    x_train, x_test, y_train, y_test, W_train, W_test = shuffle_and_split(x, y, test_size = 0.3, quadrature_weights = W, shuffle = False)

    n_train_expected = int((1 - 0.3) * x.shape[0])
    assert torch.allclose(x_train, x[:n_train_expected])
    assert torch.allclose(x_test, x[n_train_expected:])
    assert torch.allclose(y_train, y[:n_train_expected])
    assert torch.allclose(y_test, y[n_train_expected:])
    assert W_train is not None
    assert W_test is not None
    assert torch.allclose(W_train, W[:n_train_expected])
    assert torch.allclose(W_test, W[n_train_expected:])

def test_shuffle_and_split_x_y_shape_check() -> None:
    '''Tests that shuffle_and_split raises when x and y have different sample counts.'''
    x = torch.randn(10, 3)
    y = torch.randn(9, 2)

    with pytest.raises(ValueError):
        shuffle_and_split(x, y)

def test_shuffle_and_split_quadrature_weights_shape_check() -> None:
    '''Tests that shuffle_and_split raises when quadrature_weights length does not match sample count.'''
    x = torch.randn(10, 3)
    y = torch.randn(10, 2)
    W = torch.randn(9)

    with pytest.raises(ValueError):
        shuffle_and_split(x, y, quadrature_weights = W)

def test_force_h() -> None:
    '''Tests whether force_h correctly eliminates spurious imaginary parts in a matrix that is theoretically required to be Hermitian.'''
    A = torch.tensor([[1., 2.], [2., 3.]]) + 1j*torch.tensor([[1e-10, 1e-10], [1e-10, 1e-10]])
    A_h = force_h(A)
    
    assert torch.allclose(A_h.imag, torch.zeros_like(A.imag), atol = 1e-9)
    assert torch.allclose(A_h.real, A.real)

def test_force_h_shape_check() -> None:
    '''Tests whether force_h raises an error if the input is not a square matrix.'''
    A = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    
    with pytest.raises(ValueError):
        force_h(A)

def test_force_h_vectorized() -> None:
    '''Tests whether force_h_vectorized correctly eliminates spurious imaginary parts in a batch of matrices that are theoretically required to be Hermitian.'''
    A = torch.tensor([[[1., 2.], [2., 3.]], [[4., 5.], [5., 6.]]]) + 1j*torch.tensor([[[1e-10, 1e-10], [1e-10, 1e-10]], [[1e-10, 1e-10], [1e-10, 1e-10]]])
    A_h = force_h_vectorized(A)
    
    assert torch.allclose(A_h.imag, torch.zeros_like(A.imag), atol = 1e-9)
    assert torch.allclose(A_h.real, A.real)

def test_force_h_vectorized_shape_check() -> None:
    '''Tests whether force_h_vectorized raises an error if the input is not a batch of square matrices.'''
    A = torch.tensor([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]])
    
    with pytest.raises(ValueError):
        force_h_vectorized(A)