# ThesisODE2026

Python code and supporting computational files for the thesis:

**Finite-Time Blow-Up and Stopping Times for Nonlinear ODEs**  
Matthew Clark Jr.  
Department of Mathematics  
Louisiana State University  
Fall 2026

This repository contains the numerical experiments and supporting computational materials used in the thesis. The code supports the discussion of Lotka–Volterra dynamics, coordinate-wise Lie–Trotter and Strang splitting, stopping times, finite-time blow-up, and the Riccati linearization used to compute maximal forward lifetimes.

## Repository Files

- `lotka_volterra_lab.py` — runs the Lotka–Volterra numerical laboratory, including Forward Euler, coordinate-wise Lie–Trotter splitting, coordinate-wise Strang splitting, and a `solve_ivp` reference solution.
- `LVinput.txt` — input file used by `lotka_volterra_lab.py`. It supplies the variables, initial values, parameter values, right-hand-side expressions, total simulation time, and number of time steps used by the Lotka–Volterra experiment. Keep this file in the same directory as `lotka_volterra_lab.py`.
- `riccati_stopping_time.py` — computes the maximal forward lifetime for the Riccati problem by solving the associated linear second-order equation and detecting the first zero with `solve_ivp`. It also generates the linear zero-detection and reconstructed Riccati blow-up figures used in Chapter 4.
- `stop_time_stream_plots.py` — generates the Chapter 4 visualizations for Example 3 and Example 4. Example 3 illustrates discontinuity of the stopping-time function caused by exit from the prescribed state space, while Example 4 illustrates finite-time blow-up and the associated Koopman-eigenfunction viewpoint.
- `requirements.txt` — lists the Python packages required to run the numerical experiments.
- `README.md` — documentation for the repository.

## Python Requirements

The numerical experiments use:

- NumPy
- SciPy
- Matplotlib
- SymPy

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Running the Lotka–Volterra Numerical Laboratory

Make sure the following two files are in the same directory:

```text
lotka_volterra_lab.py
LVinput.txt
```

Then run:

```bash
python lotka_volterra_lab.py
```

The script reads the simulation setup from `LVinput.txt`, constructs the numerical problem, and compares:

- Forward Euler
- coordinate-wise Lie–Trotter splitting
- coordinate-wise Strang splitting
- a reference solution computed with `scipy.integrate.solve_ivp`

The numerical results are evaluated using the Hamiltonian-like observable developed in Chapter 3 of the thesis. Relative error in this observable is used as a diagnostic for comparing how well the numerical methods reproduce the qualitative structure of the continuous Lotka–Volterra system.

For reproducibility, `LVinput.txt` should remain in the repository together with `lotka_volterra_lab.py`.

## Running the Riccati Stopping-Time Experiment

Run:

```bash
python riccati_stopping_time.py
```

This script supports the numerical blow-up experiment in Chapter 4. It studies the Riccati equation

```text
u'(t) = t^2 + u(t)^2
```

through the linearized second-order equation

```text
v''(x) + x^2 v(x) = 0.
```

With the Riccati substitution

```text
u(x) = -v'(x)/v(x),
```

the first zero `xi_star` of `v` determines the maximal forward lifetime through

```text
m(s,y) = xi_star - s.
```

For the thesis example with `s = 0` and `y = 1`, the script detects the first zero numerically and generates the figures showing both the linear zero-detection problem and the reconstructed Riccati solution approaching blow-up.

## Running the Stopping-Time and Phase-Portrait Script

Run:

```bash
python stop_time_stream_plots.py
```

This script supports Examples 3 and 4 in Chapter 4.

### Example 3: Discontinuous Stopping-Time Function Caused by Domain Exit

Example 3 uses the constant vector field

```text
F(x,y) = (1,0)
```

on the state space

```text
Omega = R^2 \ Gamma,
Gamma = {(x,0) : x >= 1}.
```

Trajectories move horizontally to the right. Initial conditions with `y0 != 0` remain in the state space for all forward time, while initial conditions with `y0 = 0` and `x0 < 1` approach the removed ray in finite time. The visualization is intended to illustrate how the stopping-time function can fail to be continuous even though the trajectory itself remains bounded.

### Example 4: Finite-Time Blow-Up and a Positive Koopman Eigenfunction

Example 4 uses the autonomous system

```text
x' = y - x^2
y' = x - y
```

and highlights the positively invariant negative quadrant used in the finite-time blow-up estimate. The visualization supports the discussion of how the maximal existence-time function can be encoded through the positive Koopman observable

```text
phi = exp(-m).
```

## Thesis Connection

The repository is intended to provide the complete executable computational materials associated with the thesis without reproducing the full source code in the thesis appendix.

The thesis body focuses on the mathematical structure of the methods and the numerical conclusions, while this repository provides the runnable scripts, supporting input file, software requirements, and instructions needed to reproduce the computational experiments.

Repository:

https://github.com/matthewxclarkjr/ThesisODE2026
