import pytest
import torch

from pyresdmd.compute.hankel import hankel_matrices

def test_hankel_identity() -> None:
    '''Tests that hankel_matrices produces correct output for identity.'''
    x = torch.arange(6, dtype=torch.float32)  # trajectory of length 6
    M, N = 3, 3  # dictionary size and trajectory length
    def g(t: torch.Tensor) -> torch.Tensor:
        return t  # identity observable

    PX, PY = hankel_matrices(g, x, M, N)

    expected_PX = torch.tensor([[0., 1., 2.],
                                [1., 2., 3.],
                                [2., 3., 4.]])
    expected_PY = torch.tensor([[1., 2., 3.],
                                [2., 3., 4.],
                                [3., 4., 5.]])

    assert torch.allclose(PX, expected_PX)
    assert torch.allclose(PY, expected_PY)

def test_hankel_nontrivial() -> None:
    '''Tests that hankel_matrices produces correct output for a nontrivial observable.'''
    x = torch.arange(6, dtype=torch.float32)  # trajectory of length 6
    M, N = 3, 3  # dictionary size and trajectory length
    def g(t: torch.Tensor) -> torch.Tensor:
         return torch.exp(t)*t + 4/(1 + t**2)
    PX, PY = hankel_matrices(g, x, M, N)
    expected_PX = torch.tensor([[g(torch.tensor(0.)), g(torch.tensor(1.)), g(torch.tensor(2.))],
                                [g(torch.tensor(1.)), g(torch.tensor(2.)), g(torch.tensor(3.))],
                                [g(torch.tensor(2.)), g(torch.tensor(3.)), g(torch.tensor(4.))]])
    expected_PY = torch.tensor([[g(torch.tensor(1.)), g(torch.tensor(2.)), g(torch.tensor(3.))],
                                [g(torch.tensor(2.)), g(torch.tensor(3.)), g(torch.tensor(4.))],
                                [g(torch.tensor(3.)), g(torch.tensor(4.)), g(torch.tensor(5.))]])
    assert torch.allclose(PX, expected_PX)
    assert torch.allclose(PY, expected_PY)

@pytest.mark.parametrize("x", ['bananas', 123, [1, 2, 3], (1, 2, 3), None])
def test_hankel_x_not_tensor(x: object) -> None:
    '''Tests that hankel_matrices raises TypeError when x is not a torch.Tensor.'''
    M, N = 3, 3
    def g(t: torch.Tensor) -> torch.Tensor:
        return t

    with pytest.raises(TypeError):
        hankel_matrices(g, x, M, N)

def test_hankel_m_n_not_integer() -> None:
    '''Tests that hankel_matrices raises TypeError when M or N is not an integer.'''
    x = torch.arange(6, dtype=torch.float32)
    def g(t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(t)

    with pytest.raises(TypeError):
        hankel_matrices(g, x, "3", 3)

    with pytest.raises(TypeError):
        hankel_matrices(g, x, 3, "3")

def test_hankel_m_n_not_positive() -> None:
    '''Tests that hankel_matrices raises ValueError when M or N is not positive.'''
    x = torch.arange(6, dtype=torch.float32)
    def g(t: torch.Tensor) -> torch.Tensor:
        return t

    with pytest.raises(ValueError):
        hankel_matrices(g, x, -1, 3)

    with pytest.raises(ValueError):
        hankel_matrices(g, x, 3, 0)

def test_hankel_wrong_input_shape() -> None:
    '''Tests that hankel_matrices raises ValueError when x has incorrect shape.'''
    x = torch.arange(5, dtype=torch.float32)  # length 5
    M, N = 3, 3  # expected length M + N = 6
    def g(t: torch.Tensor) -> torch.Tensor:
        return torch.sin(t)

    with pytest.raises(ValueError):
        hankel_matrices(g, x, M, N)

@pytest.mark.parametrize("shape", [(6, 3), (4, 5), (1, 1), (8, 2)])
def test_hankel_wrong_output_shape(shape: tuple[int, int]) -> None:
    '''Tests that hankel_matrices raises ValueError when g produces output of incorrect shape.'''
    M, N = shape
    x = torch.arange(M + N, dtype=torch.float32)
    def g(t: torch.Tensor) -> torch.Tensor:
        return t
    PX, PY = hankel_matrices(g, x, M, N)

    assert PX.shape == (M, N)
    assert PY.shape == (M, N)

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hankel_dtype(dtype: torch.dtype) -> None:
    '''Tests that hankel_matrices preserves dtype'''
    x = torch.arange(8, dtype = dtype)
    M, N = 4, 4
    def g(t: torch.Tensor) -> torch.Tensor:
        return torch.exp(t)
    PX, PY = hankel_matrices(g, x, M, N)
    
    assert PX.dtype == dtype 
    assert PY.dtype == dtype 

@pytest.mark.parametrize("device", [torch.device('cpu'), torch.device('cuda:0')] if torch.cuda.is_available() else [torch.device('cpu')])
def test_hankel_device(device: torch.device) -> None:
    '''Tests that hankel_matrices preserves device'''
    x = torch.arange(8, device = device)
    M, N = 4, 4
    def g(t: torch.Tensor) -> torch.Tensor:
        return torch.cos(t)
    PX, PY = hankel_matrices(g, x, M, N)
    
    assert PX.device == device 
    assert PY.device == device

def test_hankel_non_vectorized_g() -> None:
    '''Tests that hankel_matrices raises an error when g produces a value of the wrong shape.'''
    x = torch.arange(6, dtype=torch.float32)
    M, N = 3, 3
    def g(t: torch.Tensor) -> torch.Tensor:
        return t[:-1]  # produces shape (5,) instead of (6,)

    with pytest.raises(ValueError):
        hankel_matrices(g, x, M, N)