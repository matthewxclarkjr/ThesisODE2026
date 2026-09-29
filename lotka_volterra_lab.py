import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sympy import symbols, sympify, lambdify


# Suppress warnings for cleaner output.
warnings.filterwarnings("ignore", category=UserWarning)


# Allowed functions for safe parsing with sympy.
allowed_functions = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "abs": np.abs,
}


def read_input_file(filename):
    """
    Reads an input file of the form:

        2
        x 10
        y 5
        alpha 1.5
        beta 1.0
        delta 1.0
        tau 3.0
        alpha*x - beta*x*y
        delta*x*y - tau*y
        30 3000

    The first line is the number of equations.
    The next lines give variable names and initial values.
    Then come parameter names and values.
    Then come the right-hand sides of the equations.
    The final line gives total time and number of steps.
    """

    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Remove comments and empty lines.
    lines = [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]

    idx = 0

    # Number of equations.
    num_eqs = int(lines[idx])
    idx += 1

    # Variables and initial values.
    var_names = []
    init_vals = []
    for _ in range(num_eqs):
        var_line = lines[idx]
        idx += 1
        var_name, init_val = var_line.split()
        var_names.append(var_name)
        init_vals.append(float(init_val))

    # Parameters and their values.
    # This assumes that the parameter section ends when the equation section begins.
    param_values = {}
    while idx < len(lines):
        parts = lines[idx].split()

        # If the line does not look like "name value", then it is probably an equation.
        if len(parts) != 2:
            break

        # If the second part cannot be converted to a float, it is not a parameter line.
        try:
            param_val = float(parts[1])
        except ValueError:
            break

        param_name = parts[0]
        param_values[param_name] = param_val
        idx += 1

    # Equations, one for each variable.
    equations = []
    for _ in range(num_eqs):
        eq_line = lines[idx]
        idx += 1
        equations.append(eq_line)

    # Time settings.
    t_settings = lines[idx]
    t_total, n_steps = map(float, t_settings.split())
    n_steps = int(n_steps)

    return num_eqs, var_names, init_vals, param_values, equations, t_total, n_steps


def H(x, y, alpha, beta, delta, tau):
    """
    Hamiltonian-like first integral for the Lotka-Volterra system:

        H(x,y) = delta*x - tau*log(x) + beta*y - alpha*log(y)

    This function is defined for x > 0 and y > 0.
    """

    return delta * x - tau * np.log(x) + beta * y - alpha * np.log(y)


def relative_error_in_H(x_vals, y_vals, H0, alpha, beta, delta, tau):
    """
    Computes relative error in the Hamiltonian-like invariant H:

        100 * (H(x_n,y_n) - H0) / H0

    Returned as a percentage.
    """

    H_vals = H(x_vals, y_vals, alpha, beta, delta, tau)
    return 100.0 * (H_vals - H0) / H0


def explicit_euler_solve(init_vals, equation_funcs, var_names, param_values, t_eval, dt):
    """
    Solves the system using the explicit Euler method:

        z_{n+1} = z_n + dt*f(z_n).
    """

    n_steps = len(t_eval) - 1
    num_eqs = len(var_names)

    results = {
        var: np.zeros(n_steps + 1)
        for var in var_names
    }

    # Set initial values.
    for i, var in enumerate(var_names):
        results[var][0] = init_vals[i]

    # Euler integration.
    for k in range(n_steps):
        current_vars = {
            var_names[i]: results[var_names[i]][k]
            for i in range(num_eqs)
        }

        args = [current_vars[var] for var in var_names] + list(param_values.values())

        for i in range(num_eqs):
            increment = equation_funcs[i](*args) * dt
            results[var_names[i]][k + 1] = current_vars[var_names[i]] + increment

    return results


def solve_ivp_reference(init_vals, equation_funcs, var_names, param_values, t_total, t_eval):
    """
    Solves the system using scipy.integrate.solve_ivp as a reference method.
    """

    def system_of_eqs(t, z):
        current_vars = dict(zip(var_names, z))
        args = [current_vars[var] for var in var_names] + list(param_values.values())
        dzdt = [func(*args) for func in equation_funcs]
        return dzdt

    sol = solve_ivp(
        system_of_eqs,
        [0, t_total],
        init_vals,
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-12,
    )

    ode_results = {
        var_names[i]: sol.y[i]
        for i in range(len(var_names))
    }

    return ode_results


# -----------------------------------------------------------------------------
# Coordinate-wise partial flows for Lotka-Volterra splitting
# -----------------------------------------------------------------------------


def lv_sigma(x, y, h, alpha, beta):
    """
    Advance x for time h while holding y fixed:

        x' = x * (alpha - beta*y)

    This is the one-dimensional partial flow

        sigma(h, x; y) = x * exp(h * (alpha - beta*y)).
    """

    return x * np.exp(h * (alpha - beta * y))


def lv_gamma(y, x, h, delta, tau):
    """
    Advance y for time h while holding x fixed:

        y' = y * (delta*x - tau)

    This is the one-dimensional partial flow

        gamma(h, y; x) = y * exp(h * (delta*x - tau)).
    """

    return y * np.exp(h * (delta * x - tau))


def lv_lie_step(z, h, alpha, beta, delta, tau):
    """
    One coordinate-wise Lie-Trotter splitting step.

    First freeze y and advance x:

        x_{n+1} = sigma(h, x_n; y_n).

    Then freeze the updated x_{n+1} and advance y:

        y_{n+1} = gamma(h, y_n; x_{n+1}).
    """

    x, y = z

    x_new = lv_sigma(x, y, h, alpha, beta)
    y_new = lv_gamma(y, x_new, h, delta, tau)

    return np.array([x_new, y_new], dtype=float)


def lv_strang_step(z, h, alpha, beta, delta, tau):
    """
    One coordinate-wise Strang splitting step.

    The symmetric sequence is:

        x half-step -> y full-step -> x half-step.
    """

    x, y = z

    # Half-step in x while y is frozen.
    x_half = lv_sigma(x, y, h / 2.0, alpha, beta)

    # Full-step in y while x_half is frozen.
    y_new = lv_gamma(y, x_half, h, delta, tau)

    # Second half-step in x while y_new is frozen.
    x_new = lv_sigma(x_half, y_new, h / 2.0, alpha, beta)

    return np.array([x_new, y_new], dtype=float)


def splitting_solve(z0, h, n_steps, alpha, beta, delta, tau, method="lie"):
    """
    Evolve the Lotka-Volterra system using either coordinate-wise
    Lie-Trotter or Strang splitting.
    """

    traj = np.zeros((n_steps + 1, 2))
    traj[0, :] = np.array(z0, dtype=float)
    z = np.array(z0, dtype=float)

    method = method.lower().strip()

    for k in range(n_steps):
        if method == "lie":
            z = lv_lie_step(z, h, alpha, beta, delta, tau)
        elif method == "strang":
            z = lv_strang_step(z, h, alpha, beta, delta, tau)
        else:
            raise ValueError("method must be either 'lie' or 'strang'")

        traj[k + 1, :] = z

    return traj


def build_equation_functions(var_names, param_values, equations):
    """
    Builds numerical right-hand-side functions using SymPy and lambdify.
    """

    variable_symbols = symbols(var_names)
    parameter_symbols = symbols(list(param_values.keys()))

    locals_dict = dict(
        zip(
            var_names + list(param_values.keys()),
            list(variable_symbols) + list(parameter_symbols),
        )
    )

    equation_funcs = []
    for expr_str in equations:
        expr = sympify(expr_str, locals=locals_dict)
        func = lambdify(
            variable_symbols + parameter_symbols,
            expr,
            modules=[allowed_functions, "numpy"],
        )
        equation_funcs.append(func)

    return equation_funcs


def make_plots(
    var_names,
    init_vals,
    param_values,
    t_total,
    n_steps,
    t_eval,
    dt,
    euler_results,
    lie_results,
    strang_results,
    ode_results,
    relative_error_euler,
    relative_error_lie,
    relative_error_strang,
    relative_error_ode,
    H0,
):
    """
    Generates the visualizations used in the Lotka-Volterra numerical laboratory.
    """

    alpha = param_values["alpha"]
    beta = param_values["beta"]
    delta = param_values["delta"]
    tau = param_values["tau"]
    x0 = init_vals[0]
    y0 = init_vals[1]

    # First visualization: text summary.
    fig1 = plt.figure(figsize=(11, 7))
    plt.axis("off")
    equation_text = (
        "Lotka-Volterra Equations with Parameters:\n\n"
        f"dx/dt = {alpha} * x - {beta} * x * y\n"
        f"dy/dt = {delta} * x * y - {tau} * y\n\n"
        f"Initial condition: x(0) = {x0}, y(0) = {y0}\n"
        f"Simulation Time: {t_total} units\n"
        f"Time Steps: {n_steps}\n"
        f"Step Size: h = {dt:.8f}\n\n"
        "H(x,y) = delta*x - tau*log(x) + beta*y - alpha*log(y)\n"
        f"H(x0,y0) = {H0:.10f}\n\n"
        "Final Relative Error in H:\n"
        f"  Explicit Euler: {relative_error_euler[-1]:.10f}%\n"
        f"  Lie Splitting:  {relative_error_lie[-1]:.10f}%\n"
        f"  Strang Splitting: {relative_error_strang[-1]:.10f}%\n"
        f"  ODE Solver:     {relative_error_ode[-1]:.10f}%"
    )
    plt.text(
        0.05,
        0.5,
        equation_text,
        fontsize=12,
        family="monospace",
        verticalalignment="center",
        horizontalalignment="left",
    )
    plt.title("Lotka-Volterra Parameters and Final Relative Errors in H", fontsize=14)
    plt.tight_layout()
    fig1.savefig("lv_parameter_summary.png", dpi=300)
    plt.close(fig1)

    # Second visualization: relative error comparison.
    fig2 = plt.figure(figsize=(11, 7))
    plt.plot(t_eval, relative_error_euler, label="Explicit Euler")
    plt.plot(t_eval, relative_error_lie, label="Lie Splitting")
    plt.plot(t_eval, relative_error_strang, label="Strang Splitting")
    plt.plot(t_eval, relative_error_ode, label="solve_ivp Reference")
    plt.xlabel("Time t")
    plt.ylabel("Relative Error in H [%]")
    plt.title("Relative Error Comparison for Hamiltonian-like Invariant H")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig2.savefig("lv_relative_error.png", dpi=300)
    plt.close(fig2)

    # Third visualization: variable comparison.
    fig3 = plt.figure(figsize=(11, 7))
    for var in var_names:
        plt.plot(t_eval, euler_results[var], "--", label=f"{var}(t), Explicit Euler")
        plt.plot(t_eval, lie_results[var], "-.", label=f"{var}(t), Lie Splitting")
        plt.plot(t_eval, strang_results[var], ":", label=f"{var}(t), Strang Splitting")
        plt.plot(t_eval, ode_results[var], "-", label=f"{var}(t), solve_ivp Reference")
    plt.xlabel("Time t")
    plt.ylabel("Population values")
    plt.title("Population Variables for Euler, Lie, Strang, and solve_ivp")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    fig3.savefig("lv_time_series.png", dpi=300)
    plt.close(fig3)

    # Fourth visualization: phase portrait.
    x_name = var_names[0]
    y_name = var_names[1]
    fig4 = plt.figure(figsize=(11, 7))
    plt.plot(euler_results[x_name], euler_results[y_name], "--", label="Explicit Euler")
    plt.plot(lie_results[x_name], lie_results[y_name], "-.", label="Lie Splitting")
    plt.plot(strang_results[x_name], strang_results[y_name], ":", label="Strang Splitting")
    plt.plot(ode_results[x_name], ode_results[y_name], "-", label="solve_ivp Reference")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Phase Plot: Euler, Lie, Strang, and solve_ivp")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig4.savefig("lv_phase_plot.png", dpi=300)
    plt.close(fig4)

    print("Saved plots:")
    print("  lv_parameter_summary.png")
    print("  lv_relative_error.png")
    print("  lv_time_series.png")
    print("  lv_phase_plot.png")


def run_lotka_volterra_lab(filename="LVinput.txt", make_figures=True):
    """
    Runs the full Lotka-Volterra numerical laboratory.
    """

    num_eqs, var_names, init_vals, param_values, equations, t_total, n_steps = read_input_file(filename)

    if num_eqs != 2:
        raise ValueError("This Lotka-Volterra splitting code expects exactly two variables.")

    required_params = ["alpha", "beta", "delta", "tau"]
    for param in required_params:
        if param not in param_values:
            raise ValueError(f"Required parameter '{param}' not found in input file.")

    alpha = param_values["alpha"]
    beta = param_values["beta"]
    delta = param_values["delta"]
    tau = param_values["tau"]

    t_eval = np.linspace(0, t_total, n_steps + 1)
    dt = t_total / n_steps

    equation_funcs = build_equation_functions(var_names, param_values, equations)

    # Run numerical methods.
    euler_results = explicit_euler_solve(
        init_vals,
        equation_funcs,
        var_names,
        param_values,
        t_eval,
        dt,
    )

    ode_results = solve_ivp_reference(
        init_vals,
        equation_funcs,
        var_names,
        param_values,
        t_total,
        t_eval,
    )

    lie_traj = splitting_solve(
        init_vals,
        dt,
        n_steps,
        alpha,
        beta,
        delta,
        tau,
        method="lie",
    )
    lie_results = {
        var_names[0]: lie_traj[:, 0],
        var_names[1]: lie_traj[:, 1],
    }

    strang_traj = splitting_solve(
        init_vals,
        dt,
        n_steps,
        alpha,
        beta,
        delta,
        tau,
        method="strang",
    )
    strang_results = {
        var_names[0]: strang_traj[:, 0],
        var_names[1]: strang_traj[:, 1],
    }

    # Hamiltonian-like invariant and relative error computation.
    x0 = init_vals[0]
    y0 = init_vals[1]
    H0 = H(x0, y0, alpha, beta, delta, tau)

    relative_error_euler = relative_error_in_H(
        euler_results[var_names[0]],
        euler_results[var_names[1]],
        H0,
        alpha,
        beta,
        delta,
        tau,
    )
    relative_error_lie = relative_error_in_H(
        lie_results[var_names[0]],
        lie_results[var_names[1]],
        H0,
        alpha,
        beta,
        delta,
        tau,
    )
    relative_error_strang = relative_error_in_H(
        strang_results[var_names[0]],
        strang_results[var_names[1]],
        H0,
        alpha,
        beta,
        delta,
        tau,
    )
    relative_error_ode = relative_error_in_H(
        ode_results[var_names[0]],
        ode_results[var_names[1]],
        H0,
        alpha,
        beta,
        delta,
        tau,
    )

    print("Lotka-Volterra Parameters and Final Relative Errors in H")
    print("--------------------------------------------------------")
    print(f"dx/dt = {alpha} * x - {beta} * x * y")
    print(f"dy/dt = {delta} * x * y - {tau} * y")
    print()
    print(f"Initial condition: x(0) = {x0}, y(0) = {y0}")
    print(f"Simulation Time: {t_total} units")
    print(f"Time Steps: {n_steps}")
    print(f"Step Size: h = {dt:.8f}")
    print()
    print("H(x,y) = delta*x - tau*log(x) + beta*y - alpha*log(y)")
    print(f"H(x0,y0) = {H0:.10f}")
    print()
    print("Final Relative Error in H:")
    print(f"  Explicit Euler: {relative_error_euler[-1]:.10f}%")
    print(f"  Lie Splitting:  {relative_error_lie[-1]:.10f}%")
    print(f"  Strang Splitting: {relative_error_strang[-1]:.10f}%")
    print(f"  ODE Solver:     {relative_error_ode[-1]:.10f}%")

    if make_figures:
        make_plots(
            var_names,
            init_vals,
            param_values,
            t_total,
            n_steps,
            t_eval,
            dt,
            euler_results,
            lie_results,
            strang_results,
            ode_results,
            relative_error_euler,
            relative_error_lie,
            relative_error_strang,
            relative_error_ode,
            H0,
        )

    return {
        "t_eval": t_eval,
        "dt": dt,
        "H0": H0,
        "euler_results": euler_results,
        "lie_results": lie_results,
        "strang_results": strang_results,
        "ode_results": ode_results,
        "relative_error_euler": relative_error_euler,
        "relative_error_lie": relative_error_lie,
        "relative_error_strang": relative_error_strang,
        "relative_error_ode": relative_error_ode,
    }


if __name__ == "__main__":
    run_lotka_volterra_lab("LVinput.txt", make_figures=True)
