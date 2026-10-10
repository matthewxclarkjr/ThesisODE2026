import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def linearized_riccati_system(x, V):
    """
    First-order system equivalent to

        v''(x) + x^2 v(x) = 0.

    With p = v', we have

        v' = p,
        p' = -x^2 v.
    """
    v, p = V
    return [p, -(x**2) * v]


def first_zero_event(x, V):
    """
    Event function for detecting a zero of v.
    """
    return V[0]


# Stop at the first detected zero of v.
first_zero_event.terminal = True
first_zero_event.direction = 0


def riccati_blowup_time(
    s,
    y0,
    xmax_offset=20.0,
    rtol=1e-9,
    atol=1e-11,
    max_step=0.05,
):
    """
    Compute the maximal forward lifetime m(s,y0) for

        u'(t) = t^2 + u(t)^2,   u(s) = y0,

    by solving the linearized equation

        v''(x) + x^2 v(x) = 0,
        v(s) = 1,
        v'(s) = -y0,

    and detecting the first zero xi_star of v to the right of s.

    Since

        u(x) = -v'(x) / v(x),

    the Riccati solution blows up when v first reaches zero. Therefore

        m(s,y0) = xi_star - s.

    Returns
    -------
    m : float
        Maximal forward lifetime.
    xi_star : float
        Physical blow-up location.
    sol : scipy.integrate.OdeResult
        Numerical solution object returned by solve_ivp.
    """
    s = float(s)
    y0 = float(y0)

    sol = solve_ivp(
        linearized_riccati_system,
        (s, s + xmax_offset),
        [1.0, -y0],
        events=first_zero_event,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=True,
    )

    if len(sol.t_events[0]) == 0:
        raise RuntimeError(
            "No zero of v was detected. Increase xmax_offset."
        )

    xi_star = float(sol.t_events[0][0])
    m = xi_star - s

    return m, xi_star, sol


def sample_linear_solution(sol, s, xi_star, num_points=1200):
    """
    Sample v and p=v' on [s, xi_star].
    """
    xs = np.linspace(float(s), float(xi_star), int(num_points))
    vals = sol.sol(xs)
    v_vals = vals[0]
    p_vals = vals[1]
    return xs, v_vals, p_vals


def sample_reconstructed_riccati(
    sol,
    s,
    xi_star,
    num_points=1200,
    endpoint_fraction=0.999,
):
    """
    Reconstruct

        u(x) = -v'(x)/v(x)

    on an interval ending just before xi_star.

    The endpoint is kept slightly to the left of xi_star so that division
    by v(x) remains finite in the numerical plot.
    """
    if not (0.0 < endpoint_fraction < 1.0):
        raise ValueError("endpoint_fraction must lie strictly between 0 and 1.")

    s = float(s)
    xi_star = float(xi_star)
    x_end = s + endpoint_fraction * (xi_star - s)

    xs = np.linspace(s, x_end, int(num_points))
    vals = sol.sol(xs)
    v_vals = vals[0]
    p_vals = vals[1]

    u_vals = -p_vals / v_vals
    return xs, u_vals


def plot_linear_zero(
    s=0.0,
    y0=1.0,
    xmax_offset=20.0,
    save_path="riccati_linear_zero.png",
):
    """
    Plot the linearized solution v(x) and mark its first zero xi_star.
    """
    m, xi_star, sol = riccati_blowup_time(
        s=s,
        y0=y0,
        xmax_offset=xmax_offset,
    )

    xs, v_vals, _ = sample_linear_solution(
        sol,
        s=s,
        xi_star=xi_star,
    )

    plt.figure(figsize=(10, 6))
    plt.plot(xs, v_vals, label=r"$v(x)$")
    plt.axhline(0.0, linestyle="--", label=r"$v=0$")
    plt.axvline(
        xi_star,
        linestyle=":",
        label=rf"$\xi_* \approx {xi_star:.10f}$",
    )
    plt.plot(
        [xi_star],
        [0.0],
        marker="o",
        linestyle="None",
        label="First zero",
    )
    plt.xlabel(r"$x$")
    plt.ylabel(r"$v(x)$")
    plt.title(
        "Linearized Riccati Equation: First Zero of $v$"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return m, xi_star


def plot_reconstructed_riccati(
    s=0.0,
    y0=1.0,
    xmax_offset=20.0,
    endpoint_fraction=0.999,
    save_path="riccati_reconstructed_blowup.png",
):
    """
    Plot the reconstructed Riccati solution

        u(x) = -v'(x)/v(x)

    up to a point just before the first zero of v.
    """
    m, xi_star, sol = riccati_blowup_time(
        s=s,
        y0=y0,
        xmax_offset=xmax_offset,
    )

    xs, u_vals = sample_reconstructed_riccati(
        sol,
        s=s,
        xi_star=xi_star,
        endpoint_fraction=endpoint_fraction,
    )

    plt.figure(figsize=(10, 6))
    plt.plot(xs, u_vals, label=r"$u(x)=-v'(x)/v(x)$")
    plt.axvline(
        xi_star,
        linestyle=":",
        label=rf"$\xi_* \approx {xi_star:.10f}$",
    )
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(x)$")
    plt.title(
        "Reconstructed Riccati Solution Approaching Blow-up"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return m, xi_star


def run_example():
    """
    Run the thesis example

        u'(t) = t^2 + u(t)^2,
        u(0) = 1.

    For s=0, the physical blow-up location xi_star and the maximal
    forward lifetime m(0,1) are numerically equal.
    """
    s = 0.0
    y0 = 1.0

    m, xi_star, _ = riccati_blowup_time(s=s, y0=y0)

    print("Riccati Blow-up Time via Linearization")
    print("--------------------------------------")
    print("Equation: u'(t) = t^2 + u(t)^2")
    print(f"Initial condition: u({s}) = {y0}")
    print()
    print("Linearized problem:")
    print("  v''(x) + x^2 v(x) = 0")
    print(f"  v({s}) = 1")
    print(f"  v'({s}) = {-y0}")
    print()
    print(f"First zero xi_*:             {xi_star:.12f}")
    print(f"Maximal lifetime m(0,1):    {m:.12f}")
    print(f"Comparison estimate m<=1:   {m <= 1.0}")

    plot_linear_zero(
        s=s,
        y0=y0,
        save_path="riccati_linear_zero.png",
    )

    plot_reconstructed_riccati(
        s=s,
        y0=y0,
        save_path="riccati_reconstructed_blowup.png",
    )

    print()
    print("Saved plots:")
    print("  riccati_linear_zero.png")
    print("  riccati_reconstructed_blowup.png")


if __name__ == "__main__":
    run_example()
