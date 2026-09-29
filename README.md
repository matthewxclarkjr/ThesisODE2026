# ThesisODE2026

Python programs developed in support of my 2026 mathematics thesis on ordinary differential equations, numerical methods, operator splitting, and finite-time blow-up.

## Project Overview

This repository contains computational examples that accompany the numerical portions of the thesis. The current programs focus on:

- Lotka–Volterra systems and Hamiltonian behavior
- Numerical approximation of finite-time blow-up
- Stopping-time calculations
- Stream plots and related visualizations

## Files

### `lotka_volterra_lab.py`

Numerical laboratory for the Lotka–Volterra predator-prey system.

The program is used to study numerical solution behavior and the Hamiltonian

\[
H(x,y) = -\delta \ln(x) + \gamma x - \alpha \ln(y) + \beta y.
\]

The code supports numerical experimentation and visualization associated with the thesis discussion of operator splitting and numerical methods.

### `riccati_stopping_time.py`

Numerical investigation of finite-time blow-up using a Riccati-type differential equation.

The program approximates stopping times by tracking when the numerical solution reaches a prescribed threshold.

### `stop_time_stream_plots.py`

Produces stream plots and related visualizations for the stopping-time and blow-up analysis.

### `requirements.txt`

Lists the Python packages required to run the programs in this repository.

## Thesis Context

The thesis is organized around three main areas:

1. Fundamental existence and uniqueness results for ordinary differential equations
2. Linear semigroups, operator splitting, and numerical solutions
3. Finite-time blow-up and stopping times

These Python programs provide computational support for the examples and numerical experiments developed in the second and third sections.

## Requirements

The programs use Python 3.

To install the required Python packages, open a terminal in the repository folder and run:

```bash
pip install -r requirements.txt
```

The current requirements file includes:

- NumPy
- SciPy
- Matplotlib
- SymPy

## Running the Programs

After installing the required packages, run a script from the repository folder with Python.

For example:

```bash
python lotka_volterra_lab.py
```

```bash
python riccati_stopping_time.py
```

```bash
python stop_time_stream_plots.py
```

## Purpose

This repository is intended primarily as a companion to the thesis and as a reproducible record of the numerical experiments used in the project.

## Author

Matthew Clark  
2026
