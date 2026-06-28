import os
import torch
import torch.nn as nn
from pyresdmd.dicts.dictionary import Dictionary
from pyresdmd.compute.spectra import (
    compute_eigendecomposition_from_weights,
    compute_residuals,
    compute_loss,
    compute_forecast_error,
    EDMD,
)
from pyresdmd.utils.helpers import shuffle_and_split

class TrainableDictionary(nn.Module):
    '''
    Implements a trainable dictionary, using ResDMD residuals as a loss.
    '''
    def __init__(self, dictionary : Dictionary, quadrature_weights : torch.Tensor = None, eps : float = 1e-8) -> None:
        '''
        Sets up trainable dictionary. 

        Parameters
        ----------------------------------
        dictionary : Dictionary 
            Initialized dictionary to train
        
        quadrature_weights : torch.Tensor
            Quadrature weights to use for EDMD computation. 
        
        Returns 
        ----------------------------------
        None, sets class attributes.
        '''
        super().__init__()
        self.dictionary = dictionary 
        self.dictionary_size = dictionary.size 
        self.eps = eps
        self.register_buffer('quadrature_weights', quadrature_weights)
    
    @property 
    def size(self) -> int:
        return self.dictionary_size
        
    @property 
    def input_dim(self) -> int:
        return self.dictionary.input_dim 
    
    def evaluate(self, x : torch.Tensor) -> torch.Tensor:
        return self.dictionary.evaluate(x)

    def _weight_tensor(self, M : int, quadrature_weights : torch.Tensor = None, device : torch.device = None, dtype : torch.dtype = None) -> torch.Tensor:
        '''
        Selects the quadrature weights to use for optimization and diagnostics.

        If no weights are provided, defaults to uniform weights 1/M.
        '''
        W = quadrature_weights if quadrature_weights is not None else self.quadrature_weights
        
        if W is None:
            W = torch.ones(M, device = device, dtype = dtype) / M
            return W

        if W.shape[0] != M:
            raise ValueError(f"Number of quadrature weights ({W.shape[0]}) is not equal to the number of snapshots ({M})")
        
        if device is not None:
            W = W.to(device)

        if dtype is not None:
            W = W.to(dtype = dtype)

        return W

    def _loss(self, x : torch.Tensor, y : torch.Tensor, quadrature_weights : torch.Tensor = None) -> torch.Tensor:
        '''
        Computes the scalar ResDMD objective used for training.
        '''
        W = self._weight_tensor(x.shape[0], quadrature_weights, device = x.device, dtype = x.dtype)
        Psi_X, Psi_Y = self.evaluate(x), self.evaluate(y)

        Lambda, V = compute_eigendecomposition_from_weights(Psi_X, Psi_Y, W)
        residuals = compute_residuals(Lambda, V, Psi_X, Psi_Y, W)

        W_sqrt = torch.sqrt(W).unsqueeze(1)
        singvals = torch.linalg.svdvals(W_sqrt * Psi_X)

        return compute_loss(singvals, residuals, self.eps)

    def report(self, x : torch.Tensor, y : torch.Tensor, verbose : bool = False, quadrature_weights : torch.Tensor = None) -> dict:
        '''
        Computes a full spectral report for the current dictionary.
        
        Returns a dictionary containing:
            - eigenvalues: The eigenvalues of the EDMD matrix
            - eigenvectors: The eigenvectors of the EDMD matrix
            - residuals: The ResDMD residuals associated with the eigenvalues
            - loss: The overall training loss
            - cond_num: The condition number of W^(1/2) Psi_X
            - forecast_error: The one-step EDMD forecast error
        '''
        W = self._weight_tensor(x.shape[0], quadrature_weights, device = x.device, dtype = x.dtype)
        Psi_X, Psi_Y = self.evaluate(x), self.evaluate(y)
        Lambda, V = compute_eigendecomposition_from_weights(Psi_X, Psi_Y, W)
        K = EDMD(Psi_X, Psi_Y, W)
        residuals = compute_residuals(Lambda, V, Psi_X, Psi_Y, W)
            
        W_sqrt = torch.sqrt(W).unsqueeze(1)
        W_sqrtPsi_X = W_sqrt * Psi_X 
        singvals = torch.linalg.svdvals(W_sqrtPsi_X)
        
        cond_num = (singvals[0])/(singvals[-1])
        loss = compute_loss(singvals, residuals, self.eps)
        forecast_error = compute_forecast_error(Psi_X, Psi_Y, K)

        return {
            'eigenvalues': Lambda, 
            'eigenvectors': V,
            'residuals': residuals,
            'loss': loss,
            'cond_num': cond_num,
            'forecast_error': forecast_error
        }

    def fit(self, x : torch.Tensor, y : torch.Tensor, quadrature_weights : torch.Tensor = None, epochs : int = 200, patience : int = 30, lr : float = 1e-3, test_size : float = 0.3, batch_size : float = 0.1, shuffle : bool = True) -> dict[str, list[float] | int | float | None]:
        '''
        Fits the dictionary to the data.
        
        Parameters
        ----------------------------------
        x : torch.Tensor
            Trajectory data for x.
        y : torch.Tensor
            Trajectory data for y.
        quadrature_weights : torch.Tensor
            Optional quadrature weights for weighted EDMD.
        epochs : int
            Maximum number of training epochs. Default 200.
        patience : int
            Early stopping patience (epochs without improvement). Default 30.
        lr : float
            Learning rate for Adam optimizer. Default 1e-3.
        test_size : float
            Fraction of data to use for validation. Default 0.3.
        batch_size : float
            Fraction of training data to use per epoch (0 < batch_size <= 1). Default 0.1.
        shuffle : bool
            If True, randomly shuffle before splitting train/test data. If False, split sequentially.
        
        Returns
        ----------------------------------
        dict
            Training history and metrics.
        '''
        if batch_size > 1:
            return ValueError("Batch size must be between 0 and 1, representing the fraction of training data to use per epoch.")
        self.to(x.device)
        self.train()

        x_train, x_test, y_train, y_test, W_train, W_test = shuffle_and_split(
            x,
            y,
            test_size = test_size,
            quadrature_weights = quadrature_weights if quadrature_weights is not None else self.quadrature_weights,
            shuffle = shuffle,
        )

        optimizer = torch.optim.Adam(self.parameters(), lr = lr)
        patience_counter = 0
        best_loss = float('inf') # this will track the best loss 
        best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()} # this will track the parameters of the best model
        best_cond_num = None 
        best_forecast_error = None

        train_losses = []
        test_losses = []
        train_cond_nums = []
        test_cond_nums = []
        train_forecast_errors = []
        test_forecast_errors = []

        for epoch in range(epochs):
            optimizer.zero_grad()

            n_batch = max(1, int(batch_size * x_train.shape[0]))

            # we want to take a mini-batch of size n_batch from the trajectory data.
            # we want to do this without replacement and we want to make sure the indices are consistent between x, y, W to make sure everything matches.
            # hence we generate a random permutation of the indices of the training data and take the first n_batch of them.
            idx = torch.randperm(x_train.shape[0], device = x_train.device)[:n_batch]
            x_batch = x_train[idx]
            y_batch = y_train[idx]
            W_batch = None if W_train is None else W_train[idx]

            loss = self._loss(x_batch, y_batch, W_batch)

            with torch.no_grad():
                train_output = self.report(x_batch, y_batch, W_batch)
                cond_num = train_output['cond_num']
                forecast_error = train_output['forecast_error']
                test_output = self.report(x_test, y_test, W_test)
                test_loss = test_output['loss']
                test_cond_num = test_output['cond_num']
                test_forecast_error = test_output['forecast_error']

            train_losses.append(loss.item())
            train_cond_nums.append(cond_num.item())
            train_forecast_errors.append(forecast_error.item())

            test_losses.append(test_loss.item())
            test_cond_nums.append(test_cond_num.item())
            test_forecast_errors.append(test_forecast_error.item())

            if test_loss.item() < best_loss: # if test loss is an improvement
                best_loss = test_loss.item()
                best_cond_num = test_cond_num.item()
                best_forecast_error = test_forecast_error.item()
                patience_counter = 0 # reset patience counter
                best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()} # save state associated with this loss.
            else:
                patience_counter += 1
                if patience_counter >= patience: # the test loss has not improved for patience_counter epochs, so we early stop.
                    break

            loss.backward()
            optimizer.step()

        self.load_state_dict(best_state) # load best model

        return {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_cond_nums': train_cond_nums,
            'test_cond_nums': test_cond_nums,
            'train_forecast_errors': train_forecast_errors,
            'test_forecast_errors': test_forecast_errors,
            'final_epoch': epoch,
            'best_loss': best_loss,
            'best_cond_num': best_cond_num,
            'best_forecast_error': best_forecast_error
        }
    
    def save_weights(self, path : str) -> None:
        '''
        Saves the dictionary parameters (and the quadrature_weights buffer, if set)
        to disk. The file format is inferred from the extension of `path`:
 
            - '.pt' / '.pth'  : native PyTorch (recommended; lossless, preserves
                                names/shapes/dtypes exactly).
            - '.safetensors'  : safetensors format (portable, no pickle; requires
                                the optional `safetensors` package).
            - '.csv'          : flat text format in long layout
                                (name, dtype, shape, flat_index, value). Human
                                readable and easy to consume from other tools, but
                                verbose and slow for large models.
 
        Note: this saves *values* only, not the dictionary architecture. To reload,
        construct a TrainableDictionary with the same dictionary, then call
        `load_weights`.
 
        Parameters
        ----------------------------------
        path : str
            Destination file path; format is chosen from its extension.
 
        Returns
        ----------------------------------
        None
        '''
        ext = os.path.splitext(path)[1].lower()
        state = self.state_dict()
 
        if ext in ('.pt', '.pth'):
            torch.save(state, path)
        elif ext == '.safetensors':
            try:
                from safetensors.torch import save_file
            except ImportError as e:
                raise ImportError(
                    "Saving to '.safetensors' requires the 'safetensors' package "
                    "(pip install safetensors)."
                ) from e
            # safetensors needs contiguous CPU tensors and does not allow shared storage.
            save_file({k: v.detach().cpu().contiguous() for k, v in state.items()}, path)
        elif ext == '.csv':
            self._save_state_dict_csv(path, state)
        else:
            raise ValueError(
                f"Unsupported file extension '{ext}'. Use one of: .pt, .pth, .safetensors, .csv"
            )
 
    def load_weights(self, path : str, map_location : torch.device | str | None = None, strict : bool = True):
        '''
        Loads dictionary weights previously written by `save_weights`. The module
        must already be constructed with the same dictionary architecture; this
        only restores parameter/buffer values into the existing tensors.
 
        Parameters
        ----------------------------------
        path : str
            Source file path; format is inferred from its extension.
        map_location : torch.device | str | None
            Device onto which tensors are loaded (forwarded to torch.load /
            safetensors). Final placement is governed by the module's own device,
            since load_state_dict copies values into existing tensors in place.
        strict : bool
            Whether to strictly enforce matching keys (see nn.Module.load_state_dict).
 
        Returns
        ----------------------------------
        torch.nn.modules.module._IncompatibleKeys
            NamedTuple with `missing_keys` and `unexpected_keys`.
        '''
        ext = os.path.splitext(path)[1].lower()
 
        if ext in ('.pt', '.pth'):
            state = torch.load(path, map_location=map_location)
        elif ext == '.safetensors':
            try:
                from safetensors.torch import load_file
            except ImportError as e:
                raise ImportError(
                    "Loading '.safetensors' requires the 'safetensors' package "
                    "(pip install safetensors)."
                ) from e
            device = str(map_location) if map_location is not None else 'cpu'
            state = load_file(path, device=device)
        elif ext == '.csv':
            state = self._load_state_dict_csv(path)
        else:
            raise ValueError(
                f"Unsupported file extension '{ext}'. Use one of: .pt, .pth, .safetensors, .csv"
            )
 
        return self.load_state_dict(state, strict=strict)
    def forward(self, x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
        '''
        Computes the loss function at (x, y). 
        
        Parameters 
        ----------------------------------
        x : torch.Tensor 
            Trajectory data for x 
        
        y : torch.Tensor 
            Trajectory data for x 
        
        Returns 
        ----------------------------------
        torch.Tensor
            Scalar loss tensor.
        '''
        return self._loss(x, y)