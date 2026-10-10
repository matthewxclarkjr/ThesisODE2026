# ThesisODE2026

Python code and supporting computational files for the thesis:

**Finite-Time Blow-Up and Stopping Times for Nonlinear ODEs**  
Matthew Clark Jr.  
Department of Mathematics  
Louisiana State University  
Fall 2026

This repository contains the numerical experiments used to support the thesis discussion of Lotka–Volterra dynamics, coordinate-wise Lie–Trotter and Strang splitting, stopping times, and finite-time blow-up.

## Repository Files

- `lotka_volterra_lab.py` — runs the Lotka–Volterra numerical laboratory, including Explicit Euler, coordinate-wise Lie–Trotter splitting, coordinate-wise Strang splitting, and a `solve_ivp` reference solution.
- `LVinput.txt` — input file used by `lotka_volterra_lab.py`. It supplies the variables, initial values, parameter values, right-hand-side expressions, simulation time, and number of time steps used by the Lotka–Volterra experiment. Keep this file in the same directory as `lotka_volterra_lab.py`.
- `riccati_stopping_time.py` — computes the stopping time for the Riccati problem by solving the associated linear second-order equation and detecting the first zero with `solve_ivp`.
- `stop_time_stream_plots.py` — generates the stopping-time and phase-portrait visualizations used in the thesis.
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

- Explicit Euler
- coordinate-wise Lie–Trotter splitting
- coordinate-wise Strang splitting
- a reference solution computed with `scipy.integrate.solve_ivp`

The numerical results are evaluated using the Hamiltonian-like observable discussed in Chapter 3 of the thesis.

For reproducibility, `LVinput.txt` should remain in the repository together with `lotka_volterra_lab.py`.

## Running the Riccati Stopping-Time Experiment

Run:

```bash
python riccati_stopping_time.py
```

This script supports the numerical blow-up experiment in Chapter 4. It solves the linearized second-order problem associated with the Riccati equation and uses event detection to locate the first zero of the linear solution. That zero determines the maximal forward lifetime of the corresponding Riccati solution.

## Running the Stopping-Time Stream-Plot Script

Run:

```bash
python stop_time_stream_plots.py
```

This script generates the stopping-time and phase-portrait visualizations used in the thesis.

## Thesis Connection

The repository is intended to provide the complete executable computational materials associated with the thesis without reproducing the full source code in the thesis appendix.

The thesis body focuses on the mathematical structure of the methods and the numerical conclusions, while this repository provides the runnable scripts, supporting input file, software requirements, and instructions needed to reproduce the experiments.

Repository:

https://github.com/matthewxclarkjr/ThesisODE2026
