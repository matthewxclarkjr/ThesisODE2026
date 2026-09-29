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

### `stop-time-stream-plots.py`
Produces stream plots and related visualizations for the stopping-time and blow-up analysis.

> Note: This file may later be renamed `stop_time_stream_plots.py` to follow standard Python filename conventions.

## Thesis Context

The thesis is organized around three main areas:

1. Fundamental existence and uniqueness results for ordinary differential equations
2. Linear semigroups, operator splitting, and numerical solutions
3. Finite-time blow-up and stopping times

These Python programs provide computational support for the examples and numerical experiments developed in the second and third sections.

## Requirements

The programs use Python 3 and may require common scientific-computing packages such as:

```bash
pip install numpy scipy matplotlib sympy
```

Exact dependencies may vary by script.

## Running the Programs

From a terminal, navigate to the repository folder and run a script with Python:

```bash
python lotka_volterra_lab.py
```

or

```bash
python riccati_stopping_time.py
```

## Purpose

This repository is intended primarily as a companion to the thesis and as a reproducible record of the numerical experiments used in the project.

## Author

Matthew Clark  
2026
