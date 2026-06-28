import numpy as np
import time
import torch

from pyresdmd.compute.spectra import (
    compute_eigendecomposition_from_weights,
    compute_pseudospectra,
)
from pyresdmd.dicts.hermite_dictionary import HermiteDictionary
from pyresdmd.dicts import TensorProductDictionary
from pyresdmd.dicts.nn.relu.relu_dictionary import ReLUDictionary
from pyresdmd.dicts.trainable_dictionary import TrainableDictionary
from pyresdmd.utils.helpers import benchmark_metrics
from pyresdmd.utils.plotters import plot_curve, plot_pseudospec

import numpy as np
import torch
from scipy.integrate import solve_ivp


import torch


def undamped_harmonic_cts(
    initial_states: list[torch.Tensor],
    dt: float = 0.1,
    k: float = 1.0,
    steps: int = 120,
    noise: float = 0.0,
    device: str = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simulates the continuous-time undamped harmonic oscillator using its exact
    closed-form solution (no numerical integration).

    The undamped harmonic oscillator is governed by

        dx/dt = v
        dv/dt = -k * x,    where k is the "spring constant"

    Because this is a linear, time-invariant system, the exact flow over a time
    t is the constant propagator (with angular frequency omega = sqrt(k))

        [ x(t) ]   [    cos(omega t)        sin(omega t) / omega ] [ x0 ]
        [ v(t) ] = [ -omega sin(omega t)        cos(omega t)     ] [ v0 ].

    The solution is evaluated directly at the sampling times

        t = 0, dt, 2*dt, ..., steps*dt,

    so the result is exact up to floating point (no integration error, no
    energy drift) and stays on the input device / differentiable.

    It returns snapshot pairs (x, y), where y[i] is the state reached from x[i]
    after one sampling interval dt.

    Parameters
    ----------
    initial_states : list[torch.Tensor]
        List of initial states. Each state must have shape (2,) or (1, 2),
        representing [x_0, v_0].

    dt : float
        Sampling time interval. Must be positive. Default 0.1.

    k : float
        Spring constant. Must be positive. Default 1.0.

    steps : int
        Number of sampled time steps for each trajectory. Default 120.

    noise : float
        Standard deviation of optional Gaussian OBSERVATION noise added to the
        sampled states after evaluation. Default 0.0.

        Note: this is observation noise, not process noise. The dynamics are
        deterministic, so x[i] and y[i] are each the exact trajectory plus an
        independent noise draw -- unlike the old semi-implicit Euler version,
        which injected noise into the dynamics at every step.

    device : str
        Torch device for returned tensors. If None, uses the device of the
        first initial state.

    Returns
    -------
    x, y : tuple[torch.Tensor, torch.Tensor]
        Snapshot data with shape

            x.shape == (steps * len(initial_states), 2)
            y.shape == (steps * len(initial_states), 2)

        For each row i, y[i] is the sampled state one dt later than x[i].
    """

    if not initial_states:
        raise ValueError("initial_states must be a non-empty list of torch tensors")

    if dt <= 0:
        raise ValueError("dt must be positive")

    if k <= 0:
        raise ValueError("k must be positive")

    if steps < 0:
        raise ValueError("steps must be non-negative")

    if noise < 0:
        raise ValueError("noise must be non-negative")

    if device is None:
        out_device = initial_states[0].device
    else:
        out_device = torch.device(device)

    def _sanitize_state(initial_state: torch.Tensor) -> torch.Tensor:
        """
        Checks that initial_state is a torch tensor of shape (2,) or (1, 2),
        and converts it to shape (1, 2).
        """
        if not torch.is_tensor(initial_state):
            raise TypeError("Each initial state must be a torch.Tensor")

        if initial_state.ndim == 1 and initial_state.shape[0] == 2:
            state = initial_state.unsqueeze(0)
        elif initial_state.ndim == 2 and initial_state.shape == (1, 2):
            state = initial_state
        else:
            raise ValueError("Each initial state must have shape (2,) or (1, 2)")

        if not torch.is_floating_point(state):
            raise TypeError("Each initial state must use a floating-point dtype")

        return state.to(device=out_device)

    base_state = _sanitize_state(initial_states[0])
    dtype = base_state.dtype

    # Existing Euler version returns empty x and y if steps == 0.
    if steps == 0:
        empty = torch.empty((0, 2), device=out_device, dtype=dtype)
        return empty, empty

    # Stack all initial states into one tensor of shape (n_trajectories, 2).
    y0 = torch.cat(
        [_sanitize_state(initial).to(dtype=dtype) for initial in initial_states],
        dim=0,
    )

    # Evaluate the closed form in float64 for accuracy, then cast back. This
    # stays on out_device throughout (no CPU round-trip) and is differentiable.
    y0_hp = y0.to(dtype=torch.float64)
    x0 = y0_hp[:, 0:1]  # (n_trajectories, 1)
    v0 = y0_hp[:, 1:2]  # (n_trajectories, 1)

    omega = torch.sqrt(torch.tensor(k, dtype=torch.float64, device=out_device))

    # Sampling times t = 0, dt, ..., steps*dt, shape (steps + 1,).
    t = torch.arange(steps + 1, dtype=torch.float64, device=out_device) * dt
    cos_wt = torch.cos(omega * t).unsqueeze(0)  # (1, steps + 1)
    sin_wt = torch.sin(omega * t).unsqueeze(0)  # (1, steps + 1)

    # Apply the propagator. Broadcasting gives shape (n_trajectories, steps + 1).
    x_t = x0 * cos_wt + (v0 / omega) * sin_wt
    v_t = -x0 * omega * sin_wt + v0 * cos_wt

    # Shape (n_trajectories, steps + 1, 2): trajectory 0 all times, then 1, ...
    states_by_traj = torch.stack((x_t, v_t), dim=-1).to(dtype=dtype)

    if noise > 0:
        # Observation noise added after evaluation.
        states_by_traj = states_by_traj + noise * torch.randn_like(states_by_traj)

    x = states_by_traj[:, :-1, :].reshape(-1, 2)
    y = states_by_traj[:, 1:, :].reshape(-1, 2)

    return x, y

def sim(initial_states : list[torch.Tensor],
    dt : float = 0.1,
    k : float = 1.0,
    steps : int = 120,
    noise : float= 0.0,
    device : str = None) -> tuple[torch.Tensor, torch.Tensor]:
    '''
        Simulates the undamped harmonic oscillator. The undamped harmonic oscillator is governed by the coupled ODEs:
            dv/dt = -k*x, where k is the "spring constant"
            dx/dt = v
        We discretize this using semi-implicit Euler:
            x_(n + 1) = x_n + (Delta t) * v_n + eps_n
            v_(n + 1) = v_n - (Delta t) * k * x_(n + 1) + eps_n'
        where eps_n, eps_n' are optional random noise, implemented as Normal(0, noise^2).

        Parameters
        ------------------------------
        initial_states : list[torch.Tensor]
            List of initial states to evolve, typically produced by random sampling.

        dt : float 
            Time-step to use for discretization. Must be positive 

        k : float 
            The spring constant k. Default 1.0.

        steps : int 
            Number of steps to run each simulation for. Default 120. 

        noise : float 
            Noise level. Default 0.0.

        device : str 
            Device to send trajectory data to. 

        Returns
        ------------------------------
        x, y
            Output trajectory data as tensors of pairs (x_i, v_i). y[i] is the state that x[i] evolves to after one time step. 

            Has shape (total_trajectory_length, 2), where total_trajectory_length = steps * len(initial_states).
    '''
    if not initial_states:
        raise ValueError("initial_states must be a non-empty list of torch tensors")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if noise < 0:
        raise ValueError("noise must be non-negative")
    
    if device is None:
        device = initial_states[0].device

    def _sanitize_state(initial_state : torch.Tensor) -> torch.Tensor:
        '''Checks that the initial state is a torch tensor of shape (2,) or (1, 2) and converts it to shape (1, 2) if necessary.'''
        if not torch.is_tensor(initial_state):
            raise TypeError("Each initial state must be a torch.Tensor")

        # valid starting points can be shape (2,), meaning a torch tensor [x_0, v_0], or shape (1, 2), meaning a torch tensor [[x_0, v_0]]
        # we will convert the former to the latter, so that all trajectories have shape (steps, 2)

        if initial_state.ndim == 1 and initial_state.shape[0] == 2:
            state = initial_state.unsqueeze(0)
        elif initial_state.ndim == 2 and initial_state.shape == (1, 2):
            state = initial_state
        else:
            raise ValueError("Each initial state must have shape (2,) or (1, 2)")

        return state.to(device = device)

    base_state = _sanitize_state(initial_states[0])
    dtype = base_state.dtype
    device = base_state.device

    states = []

    for initial in initial_states:
        trajectory = [_sanitize_state(initial).to(dtype = dtype)]
        
        for _ in range(steps):
            state = trajectory[-1]
            x_0, v_0 = state[:, 0], state[:, 1]
            x_noise = noise * torch.randn_like(x_0)
            v_noise = noise * torch.randn_like(v_0)
            x_1 = x_0 + dt * v_0 + x_noise
            v_1 = v_0 - dt * k * x_1 + v_noise
            trajectory.append(torch.stack((x_1, v_1), dim = 1))
        states.append(torch.cat(trajectory, dim = 0))
    
    x = torch.cat([state[:-1] for state in states], dim = 0).to(device = device, dtype = dtype)
    y = torch.cat([state[1:] for state in states], dim = 0).to(device = device, dtype = dtype)

    return x, y

def udh_demo(noise : float = 0.0,
    n_trajectories : int = 300,
    steps : int = 60,
    epochs : int = 1000,
    patience : int = 50,
    dictionary_size : int = 36,
    repeats : int = 15, 
    save_plots : bool = True,
    save_report : bool = True, 
    device : str = None
) -> str:
    '''
    Replicates the Undamped Harmonic Oscillator simulation seen in our paper.

    The function runs 15 (default) independent trajectory simulations, and use this to train a ReLU dictionary. The initial states are generated by taking a number r in [0.7, 1.3] at uniform random and sampling within the circle of radius r.

    The function does the following:
    1. record the resultant loss, condition number and forecast error.
    2. computes these quantities for a Hermite dictionary. 
    3. computes a pseudospectral plot for the untrained ReLU dictionary, the trained ReLU dictionary, and the Hermite dictionary. We draw the EDMD eigenvalues as black crosses.

    If save_report = True, a .log file is saved reporting the mean and standard deviation of the loss, condition number and forecast error.

    Parameters
    ------------------------------
    noise : float 
        Noise level of simulation.

    n_trajectories : int
        Number of initial states to evolve. Default 300.

    steps : int 
        Number of steps to evolve each initial state. Default 60. 

    epochs : int 
        Number of training epochs for the ReLU dictionary. 

    patience : int 
        Number of non-improvement epochs allowed before early stopping.

    dictionary_size : int 
        Number of dictionary functions to consider for both the ReLU dictionary and the Hermite dictionary. 
        
        The Hermite dictionary will be the tensor product of two Hermite dictionaries of size sqrt(dictionary_size).

        Default 36. 

    repeats : int 
        Number of times to repeat the simulation. Default 15. 
        
    save_plots : bool 
        Whether to save plots locally. Default True. 

    save_report : bool 
        Whether to save report locally. Default True. 

    device : str 
        Device to run simulation on. Defaults to CPU if not provided.
    '''
    t = time.time()
    if not device:
        device = 'cpu'
    
    l_test_losses = []
    l_test_cond_nums = []
    l_test_forecast_errors = []
    
    l_hermite_losses = []
    l_hermite_cond_nums = []
    l_hermite_forecast_errors = []
    saved_files = []
    
    def rand_point() -> torch.Tensor:
        '''Generates a random point in the state space. We pick a radius uniformly from [0.3, 2.3] and an angle uniformly from [0, 2pi], and return the corresponding point in Cartesian coordinates.'''
        r = 2*torch.rand(1) + 0.3
        theta = 2*torch.pi*torch.rand(1)
        return torch.stack((r * torch.cos(theta), r * torch.sin(theta))).squeeze()
    
    for i in range(repeats):
        initial_states = [rand_point() for _ in range(n_trajectories)]
        base = ReLUDictionary(input_dim = 2, n_functions = dictionary_size)
        dictionary = TrainableDictionary(base)
        dictionary.to(device)
        x, y = sim(initial_states, dt = 0.5, k = 1.0, steps = steps, noise = noise, device = device)
        
        # initial stats
        report = dictionary.report(x, y)
        init_eigvals = report['eigenvalues']
        Psi_X, Psi_Y = dictionary.evaluate(x), dictionary.evaluate(y)
        init_pseudospec = compute_pseudospectra(Psi_X, Psi_Y) 
        
        # stats for trained dictionary 
        trained = dictionary.fit(x, y, epochs = epochs, patience = patience)
        train_losses = trained['train_losses']
        test_losses = trained['test_losses']
        train_cond_nums = trained['train_cond_nums']
        test_cond_nums = trained['test_cond_nums'] 
        train_forecast_errors = trained['train_forecast_errors']
        test_forecast_errors = trained['test_forecast_errors']
        final_epoch = trained['final_epoch']
        # get metrics from the model after loading the best state
        best_report = dictionary.report(x, y)
        best_loss = best_report['loss']
        best_cond_num = best_report['cond_num']
        best_forecast_error = best_report['forecast_error']
        Psi_X, Psi_Y = dictionary.evaluate(x), dictionary.evaluate(y)
        final_pseudospec = compute_pseudospectra(Psi_X, Psi_Y) 
        W = torch.ones(x.shape[0], device = x.device) / x.shape[0]
        final_eigvals, _ = compute_eigendecomposition_from_weights(Psi_X, Psi_Y, W)
        
        # untrained Hermite tensor-product dictionary (using TensorProductDictionary)
        size = int(dictionary_size**0.5)
        hermite_dict = TensorProductDictionary(HermiteDictionary(size), HermiteDictionary(size))
        hermite_dict.to(device)
        hermite_metrics = benchmark_metrics(hermite_dict, x, y)
        Lambda = hermite_metrics['Lambda']
        hermite_cond_num = hermite_metrics['cond_num']
        hermite_loss = hermite_metrics['loss']
        hermite_forecast_error = hermite_metrics['forecast_error']
        hermite_pseudospec = hermite_metrics['pseudospec']
        
        # plot curves 
        if i == 0 and save_plots:
            f1 = f'udh_sim_loss_{int(time.time())}.png'
            saved_files.append(f1)
            plot_curve(final_epoch, train_losses, test_losses, add_values = {'Hermite': hermite_loss}, 
                       displayname = 'Undamped Harmonic Oscillator', ylabel = 'Loss', save_plot = True, 
                       filename = f1)
            
            f2 = f'udh_sim_cond_num_{int(time.time())}.png'
            saved_files.append(f2)
            plot_curve(final_epoch, train_cond_nums, test_cond_nums, add_values = {'Hermite': hermite_cond_num}, 
                        displayname = 'Undamped Harmonic Oscillator', ylabel = 'Condition number', save_plot = True,
                        filename = f2)
            
            f3 = f'udh_sim_forecast_{int(time.time())}.png'
            saved_files.append(f3)
            plot_curve(final_epoch, train_forecast_errors, test_forecast_errors, add_values = {'Hermite': hermite_forecast_error}, 
                        displayname = 'Undamped Harmonic Oscillator', ylabel = 'Forecast error', save_plot = True,
                        filename = f3)
        
        # plot pseudospectra
        if i == 0 and save_plots:
            eps_levels = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1]
            re_vals = init_pseudospec['re_vals']
            im_vals = init_pseudospec['im_vals']
            tau_grid = init_pseudospec['tau_grid']
            f4 = f'udh_pseudospec_init_relu_{int(time.time())}.png'
            saved_files.append(f4)
            plot_pseudospec(re_vals, im_vals, tau_grid, eps_levels, eigvals = init_eigvals, unit_circle = True, save = True, displayname = 'Psuedospectrum of Undamped Harmonic Oscillator (Untrained ReLU)', filename = f4)
            re_vals = final_pseudospec['re_vals']
            im_vals = final_pseudospec['im_vals']
            tau_grid = final_pseudospec['tau_grid']
            f5 = f'udh_pseudospec_final_relu_{int(time.time())}.png'
            saved_files.append(f5)
            plot_pseudospec(re_vals, im_vals, tau_grid, eps_levels, eigvals = final_eigvals, unit_circle = True, save = True, displayname = 'Psuedospectrum of Undamped Harmonic Oscillator (Trained ReLU)', filename = f5)
            re_vals = hermite_pseudospec['re_vals']
            im_vals = hermite_pseudospec['im_vals']
            tau_grid = hermite_pseudospec['tau_grid']
            f6 = f'udh_pseudospec_hermite_{int(time.time())}.png'
            saved_files.append(f6)
            plot_pseudospec(re_vals, im_vals, tau_grid, eps_levels, eigvals = Lambda, unit_circle = True, save = True, displayname = 'Psuedospectrum of Undamped Harmonic Oscillator (Hermite)', filename = f6)
        # store for mean/std 
        l_test_cond_nums.append(best_cond_num)
        l_test_forecast_errors.append(best_forecast_error)
        l_test_losses.append(best_loss) 
        l_hermite_losses.append(hermite_loss)
        l_hermite_cond_nums.append(hermite_cond_num)
        l_hermite_forecast_errors.append(hermite_forecast_error)
    
    l_test_cond_nums = np.array(l_test_cond_nums)
    l_test_forecast_errors = np.array(l_test_forecast_errors)
    l_test_losses = np.array(l_test_losses)
    l_hermite_losses = np.array(l_hermite_losses)
    l_hermite_cond_nums = np.array(l_hermite_cond_nums)
    l_hermite_forecast_errors = np.array(l_hermite_forecast_errors)

    t1 = time.time()
    duration = int(t1 - t)
    report = f"""
        Ran Undamped Harmonic Oscillator simulation {repeats} times. Took {duration} seconds.
        ----------------------------------------------------------------------------
            Mean best loss for ReLU: {l_test_losses.mean()} (std: {l_test_losses.std()})
            Mean condition number for best ReLU: {l_test_cond_nums.mean()} (std: {l_test_cond_nums.std()})
            Mean forecast error for best ReLU: {l_test_forecast_errors.mean()} (std: {l_test_forecast_errors.std()})
            ---------------------------------------------------------------------------------
            Mean loss for Hermite: {l_hermite_losses.mean()} (std: {l_hermite_losses.std()})
            Mean condition number for Hermite: {l_hermite_cond_nums.mean()} (std: {l_hermite_cond_nums.std()})
            Mean forecast number for Hermite: {l_hermite_forecast_errors.mean()} (std: {l_hermite_forecast_errors.std()})
            """
    
    if save_plots:
        if saved_files:
            report += """
            ---------------------------------------------------------------------------------
            Saved files:
            """
            report += "\n" + "\n".join(f"                {filename}" for filename in saved_files)
    
    if save_report:
        with open(f"log_{int(time.time())}.log", "x") as file: 
            file.write(report)
    
    return report