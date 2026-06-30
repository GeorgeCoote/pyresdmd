# PyResDMD
Implementation of ResDMD in PyTorch, including learned dictionaries.

## Mathematical Context
Consider the discrete-time dynamical system x_(n + 1) = F(x_n). It is very difficult to deduce properties of the system directly from F, which may not even be directly accessible. Instead, we move to looking at how F transforms observables (measurements from the system) via the associated Koopman operator. 

Koopman operators are linear, allowing us apply techniques from spectral theory. Spectral properties of the Koopman operator reveal the global structure of the system in ways otherwise difficult or impossible. 

Extended Mode Decomposition (EDMD) allows us to find finite-dimensional approximations of Koopman operators, by sampling trajectories in the system, from a dictionary of observables. While EDMD approximations can perform well pointwise, they experience very serious spectral pollution, meaning that features of the system are hallucinated. 

Residual Dynamic Mode Decomposition (ResDMD: https://arxiv.org/pdf/2205.09779) was introduced by Colbrook and Townsend, allowing for the computation of "residuals", which indicate the reliability of an eigenvalue.

In this project we consider the use of a trained dictionary, a basis of observables dependent on parameters which will be determined through an optimization process. We use ResDMD residuals to form a loss function, and additionally penalise ill-conditioned models.

## Content

```
pyresdmd/
|- compute              
|  |- hankel                    - generates Hankel matrices associated with an observable
|  |- spectra                   - main file for spectral methods 
|- dicts                        - implements dictionary of observables 
|  |- nn                
|  |  |- relu                   - implements trainable dictionary based off relu activation
|  |- chebyshev_dictionary      - implements dictionary of Chebyshev polynomials
|  |- dictionary                - base class for dictionaries
|  |- fourier_mode_dictionary   - implements dictionary of Fourier modes
|  |- hermite_dictionary        - implements dictionary of Hermite functions
|  |- polynomial_dictionary     - implements dictionary of polynomials
|  |- tensor_product_dictionary - allows for multiplying multiple dictionaries
|  |- trainable_dictionary      - base class for trainable dictionaries
|- sims
|  |- duffing                   - simulates Duffing oscillator
|  |- pendulum                  - simulates nonlinear 1D pendulum
|  |- rossler                   - simulates Rossler oscillator
|  |- oisst                     - climate data simulaton
|  |- undamped_harmonic         - simulates undamped harmonic oscillator
|- examples
|  |- duffing                   - demonstrates Duffing oscillator
|  |- pendulum                  - demonstrates Pendulum
|  |- rossler                   - demonstrates Rossler
|  |- undamped_harmonic         - demonstrates Undamped Harmonic 
``` 
