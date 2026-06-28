import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from typing import Union
from time import time
from matplotlib import rc
import matplotlib.pylab as plt

def plot_curve(final_epoch : int, train_values : list[float], test_values : list[float], add_values : dict[str, float | torch.Tensor], displayname : str, ylabel : str, save_plot : bool = True, filename : str | None = None, log_yscale : bool = True) -> None:
    '''
    Plots the curve for the training and test sets, as well as any additional values provided in add_values. Saves the plot to filename.

    Parameters 
    ----------------------------------
    final_epoch : int 
        The final epoch of training, used to set the x-axis limit. 
    
    train_values : list[float] 
        A list of values on the training data at each epoch. Should have length equal to final_epoch. 
    
    test_values : list[float] 
        A list of values on the test data at each epoch. Should have length equal to final_epoch. 
    
    add_values : dict[str, float] 
        A dictionary giving the values for any deterministic benchmark. Will be plotted as a horizontal line.
    
    displayname : str 
        The display name for the main curve (train/test). This will be used in the legend.
    
    ylabel : str 
        The label for the y-axis.

    log_yscale : bool, optional
        Whether to use a logarithmic scale on the y-axis. Default is True. Note that
        non-positive values cannot be displayed on a log scale and will be clipped by
        matplotlib.
    '''
    fig = plt.figure()
    ax = fig.add_subplot()
    
    rc('font', **{'family': 'serif', 'size' : 16, 'serif': ['Computer Modern']})
    rc('text', usetex=True)

    ax.scatter(list(range(final_epoch + 1)), train_values, label = 'Training ' + ylabel)
    ax.scatter(list(range(final_epoch + 1)), test_values, label = 'Test ' + ylabel)

    for i, (label, val) in enumerate(add_values.items()):
        if isinstance(val, torch.Tensor):
            val = val.detach().cpu().numpy()
        ax.axhline(y = val, color = f'C{i}', label = label, linestyle = '--')

    if log_yscale:
        ax.set_yscale('log')

    ax.legend()
    ax.set_title(displayname)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    if save_plot:
        if not filename:
            filename = f'{displayname}_{int(time())}.png'
        fig.savefig(filename, bbox_inches = 'tight')

def plot_pseudospec(re_vals : torch.Tensor, im_vals : torch.Tensor, tau_grid : torch.Tensor, eps_levels : Union[np.ndarray, list[float]], eigvals : torch.Tensor = None, unit_circle : bool = False, save : bool = False, displayname : str = 'Pseudospectrum', filename : str | None = None) -> None:
    '''
    Plots the pseudospectrum contour plot, with optional eigenvalue overlay. Saves the plot to filename if save is True.

    Parameters 
    ----------------------------------
    re_vals : torch.Tensor 
        The values of the real part of the grid. Should be 1D.

    im_vals : torch.Tensor 
        The values of the imaginary part of the grid. Should be 1D.
    
    tau_grid : torch.Tensor
        The values of the pseudospectrum on the grid. Should have shape (len(re_vals), len(im_vals)).
    
    eps_levels : np.ndarray
        The levels of epsilon to plot for the pseudospectrum contours. Should be a list of positive floats.
    
    eigvals : torch.Tensor, optional
        The eigenvalues to overlay on the plot. Should have shape (n_eigvals, ) and be a complex tensor. Default is None, in which case no eigenvalues are plotted.
    
    save : bool, optional
        Whether to save the plot to a file. Default is False.
    
    filename : str, optional
        The filename to save the plot to, if save is True. Can include '{time}' which will be replaced by the current time. Default is f'pseudospec_{time}.png'.
    '''
    
    rc('font', **{'family': 'serif', 'size' : 16, 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    
    if not filename:
        filename = f'pseudospec_{int(time())}.png'

    eps_levels = np.array(eps_levels) 
    if (eps_levels <= 0).any():
        raise ValueError("eps_levels should be a list of positive floats")

    fig = plt.figure()
    ax = fig.add_subplot()

    tau_np = tau_grid.detach().cpu().numpy()
    re_np = re_vals.detach().cpu().numpy()
    im_np = im_vals.detach().cpu().numpy()
    tau_plot = tau_np.T

    if unit_circle:
        theta = torch.linspace(0.0, 2 * torch.pi, 512)
        unit_circle_re = torch.cos(theta).cpu().numpy()
        unit_circle_im = torch.sin(theta).cpu().numpy()
        ax.plot(
            unit_circle_re, unit_circle_im, 
            linestyle = '--',
            color = 'black',
            linewidth = 1.2
        )
    
    mesh = ax.pcolormesh(
        re_np, im_np, tau_plot, 
        cmap = 'viridis_r',
        norm = mcolors.Normalize(vmin = float(tau_np.min()), vmax = float(tau_np.max())),
        shading = 'auto',
    )

    plt.colorbar(mesh, ax = ax, label = r'$\tau_N(z)$')

    cs = ax.contour(
        re_np, im_np, tau_plot, 
        levels = sorted(eps_levels),
        colors = 'white',
        linewidths = 1.2
    )

    if eigvals is not None:
        eigvals_np = eigvals.detach().cpu().numpy()
        ax.scatter(
            eigvals_np.real, eigvals_np.imag,
            marker = 'X',
            color = 'red'
        )
    
    ax.set_xlabel(r'Re$(z)$')
    ax.set_ylabel(r'Im$(z)$')
    ax.set_title(displayname)
    ax.set_aspect('equal')

    if save:
        fig.savefig(filename, dpi = 150, bbox_inches = 'tight')

