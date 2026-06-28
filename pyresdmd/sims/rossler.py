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


def sim(initial_states : list[torch.Tensor],
    dt : float = 0.01,
    a : float = 0.2,
    b : float = 0.2,
    c : float = 5.7,
    steps : int = 400,
    noise : float = 0.0,
    burn_in : int = 0,
    device : str = None) -> tuple[torch.Tensor, torch.Tensor]:
    '''
        Simulates the R"ossler system using forward Euler discretization:
            dx/dt = -y - z
            dy/dt = x + a*y
            dz/dt = b + z*(x - c)

        The discrete update used is:
            x_{n+1} = x_n + dt * (-y_n - z_n) + noise * Normal(0,1)
            y_{n+1} = y_n + dt * (x_n + a * y_n) + noise * Normal(0,1)
            z_{n+1} = z_n + dt * (b + z_n * (x_n - c)) + noise * Normal(0,1)

        Parameters
        ------------------------------
        initial_states : list[torch.Tensor]
            List of initial states to evolve. Each can be shape (3,) or (1,3).

        dt, a, b, c, steps, noise, burn_in, device : see above.

        Returns
        ------------------------------
        x, y
            Trajectory data where `x` contains states and `y` contains the next-step states.
    '''
    if not initial_states:
        raise ValueError("initial_states must be a non-empty list of torch tensors")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if noise < 0:
        raise ValueError("noise must be non-negative")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative")

    if device is None:
        device = initial_states[0].device

    def _sanitize_state(initial_state : torch.Tensor) -> torch.Tensor:
        '''Checks that the initial state is a torch tensor of shape (3,) or (1, 3) and converts it to shape (1, 3) if necessary.'''
        if not torch.is_tensor(initial_state):
            raise TypeError("Each initial state must be a torch.Tensor")

        if initial_state.ndim == 1 and initial_state.shape[0] == 3:
            state = initial_state.unsqueeze(0)
        elif initial_state.ndim == 2 and initial_state.shape == (1, 3):
            state = initial_state
        else:
            raise ValueError("Each initial state must have shape (3,) or (1, 3)")

        return state.to(device = device)

    base_state = _sanitize_state(initial_states[0])
    dtype = base_state.dtype
    device = base_state.device

    states = []

    for initial in initial_states:
        trajectory = [_sanitize_state(initial).to(dtype = dtype)]

        for _ in range(steps):
            state = trajectory[-1]
            x_0, y_0, z_0 = state[0, 0], state[0, 1], state[0, 2]
            x_1 = x_0 + dt * (-y_0 - z_0) + noise * torch.randn(1, device = device)
            y_1 = y_0 + dt * (x_0 + a * y_0) + noise * torch.randn(1, device = device)
            z_1 = z_0 + dt * (b + z_0 * (x_0 - c)) + noise * torch.randn(1, device = device)
            trajectory.append(torch.tensor([[x_1, y_1, z_1]], device = device, dtype = dtype))

        # discard burn-in portion
        states.append(torch.cat(trajectory[burn_in:], dim = 0))

    x = torch.cat([state[:-1] for state in states], dim = 0).to(device = device, dtype = dtype)
    y = torch.cat([state[1:] for state in states], dim = 0).to(device = device, dtype = dtype)

    return x, y


def rossler_demo(noise : float = 0.0,
    n_trajectories : int = 110,
    steps : int = 400,
    epochs : int = 1000,
    patience : int = 50,
    dictionary_size : int = 81,
    burn_in : int = 0,
    grid_resolution : int = 100,
    repeats : int = 5,
    save_plots : bool = True,
    save_report : bool = True,
    device : str = None
) -> str:
    '''
    Runs the R"ossler simulation, trains a ReLU `TrainableDictionary` and benchmarks an untrained Hermite tensor-product dictionary.

    Defaults are set per request.
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
        '''Generates a random point in the state space. Samples uniformly from the box [-1, 2]^3.'''
        return 3*torch.rand(1, 3) - 1

    for i in range(repeats):
        initial_states = [rand_point() for _ in range(n_trajectories)]
        base = ReLUDictionary(input_dim = 3, n_functions = dictionary_size)
        dictionary = TrainableDictionary(base)
        dictionary.to(device)
        x, y = sim(initial_states, dt = 0.01, a = 0.2, b = 0.2, c = 5.7, steps = steps, noise = noise, burn_in = burn_in, device = device)

        # initial stats
        report = dictionary.report(x, y)
        init_eigvals = report['eigenvalues']
        Psi_X, Psi_Y = dictionary.evaluate(x), dictionary.evaluate(y)
        init_pseudospec = compute_pseudospectra(Psi_X, Psi_Y, grid_resolution = grid_resolution)

        # train
        trained = dictionary.fit(x, y, epochs = epochs, patience = patience)
        train_losses = trained['train_losses']
        test_losses = trained['test_losses']
        train_cond_nums = trained['train_cond_nums']
        test_cond_nums = trained['test_cond_nums']
        train_forecast_errors = trained['train_forecast_errors']
        test_forecast_errors = trained['test_forecast_errors']
        final_epoch = trained['final_epoch']

        best_report = dictionary.report(x, y)
        best_loss = best_report['loss']
        best_cond_num = best_report['cond_num']
        best_forecast_error = best_report['forecast_error']
        Psi_X, Psi_Y = dictionary.evaluate(x), dictionary.evaluate(y)
        final_pseudospec = compute_pseudospectra(Psi_X, Psi_Y, grid_resolution = grid_resolution)
        W = torch.ones(x.shape[0], device = x.device) / x.shape[0]
        final_eigvals, _ = compute_eigendecomposition_from_weights(Psi_X, Psi_Y, W)

        # Hermite tensor-product dictionary (3D)
        size = max(1, int(round(dictionary_size ** (1/3))))
        hermite_dict = TensorProductDictionary(HermiteDictionary(size), HermiteDictionary(size), HermiteDictionary(size))
        hermite_dict.to(device)
        hermite_metrics = benchmark_metrics(hermite_dict, x, y)
        Lambda = hermite_metrics['Lambda']
        hermite_cond_num = hermite_metrics['cond_num']
        hermite_loss = hermite_metrics['loss']
        hermite_forecast_error = hermite_metrics['forecast_error']
        hermite_pseudospec = hermite_metrics['pseudospec']

        # plot curves
        if i == 0 and save_plots:
            f1 = f'rossler_sim_loss_{int(time.time())}.png'
            saved_files.append(f1)
            plot_curve(final_epoch, train_losses, test_losses, add_values = {'Hermite': hermite_loss},
                       displayname = 'Rossler', ylabel = 'Loss', save_plot = True, filename = f1)

            f2 = f'rossler_sim_cond_num_{int(time.time())}.png'
            saved_files.append(f2)
            plot_curve(final_epoch, train_cond_nums, test_cond_nums, add_values = {'Hermite': hermite_cond_num},
                       displayname = 'Rossler', ylabel = 'Condition number', save_plot = True, filename = f2)

            f3 = f'rossler_sim_forecast_{int(time.time())}.png'
            saved_files.append(f3)
            plot_curve(final_epoch, train_forecast_errors, test_forecast_errors, add_values = {'Hermite': hermite_forecast_error},
                       displayname = 'Rossler', ylabel = 'Forecast error', save_plot = True, filename = f3)

        # plot pseudospectra
        if i == 0 and save_plots:
            eps_levels = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]
            re_vals = init_pseudospec['re_vals']
            im_vals = init_pseudospec['im_vals']
            tau_grid = init_pseudospec['tau_grid']
            f4 = f'rossler_pseudospec_init_relu_{int(time.time())}.png'
            saved_files.append(f4)
            plot_pseudospec(re_vals, im_vals, tau_grid, eps_levels, eigvals = init_eigvals, unit_circle = True, save = True, displayname = 'Psuedospectrum of Rossler (Untrained ReLU)', filename = f4)
            re_vals = final_pseudospec['re_vals']
            im_vals = final_pseudospec['im_vals']
            tau_grid = final_pseudospec['tau_grid']
            f5 = f'rossler_pseudospec_final_relu_{int(time.time())}.png'
            saved_files.append(f5)
            plot_pseudospec(re_vals, im_vals, tau_grid, eps_levels, eigvals = final_eigvals, unit_circle = True, save = True, displayname = 'Psuedospectrum of Rossler (Trained ReLU)', filename = f5)
            re_vals = hermite_pseudospec['re_vals']
            im_vals = hermite_pseudospec['im_vals']
            tau_grid = hermite_pseudospec['tau_grid']
            f6 = f'rossler_pseudospec_hermite_{int(time.time())}.png'
            saved_files.append(f6)
            plot_pseudospec(re_vals, im_vals, tau_grid, eps_levels, eigvals = Lambda, unit_circle = True, save = True, displayname = 'Psuedospectrum of Rossler (Hermite)', filename = f6)

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
        Ran Rossler simulation {repeats} times. Took {duration} seconds.
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
        with open(f"rossler_log_{int(time.time())}.log", "x") as file:
            file.write(report)

    return report