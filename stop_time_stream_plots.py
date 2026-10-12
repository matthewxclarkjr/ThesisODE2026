import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ============================================================
# Example 3:
# A discontinuous stopping-time function caused by domain exit.
#
# State space:
#   Gamma = {(x,0) : x >= 1}
#   Omega = R^2 \ Gamma
#
# Vector field:
#   F(x,y) = (1,0)
#
# Exact solution:
#   z(t;z0) = (x0 + t, y0)
#
# If y0 != 0, the trajectory never meets Gamma, so m(x0,y0)=+infinity.
# If y0 = 0 and x0 < 1, the trajectory reaches Gamma at
#   t = 1 - x0,
# so m(x0,0)=1-x0.
# ============================================================

def ex3_solution(t, z0):
    """Exact trajectory z(t;z0) = (x0+t, y0) for F=(1,0)."""
    x0, y0 = z0
    t = np.asarray(t, dtype=float)
    return x0 + t, np.full_like(t, y0, dtype=float)


def plot_example3():
    """
    Plot the constant vector field F=(1,0), the removed ray Gamma,
    one finite-lifetime trajectory on y=0, and nearby global trajectories.
    """
    x_min, x_max = -1.0, 3.0
    y_min, y_max = -1.25, 1.25

    # Constant vector field F=(1,0).
    x_grid = np.linspace(x_min, x_max, 17)
    y_grid = np.linspace(y_min, y_max, 11)
    X, Y = np.meshgrid(x_grid, y_grid)
    U = np.ones_like(X)
    V = np.zeros_like(Y)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.quiver(X, Y, U, V)

    # Removed ray Gamma = {(x,0): x >= 1}.
    gamma_x = np.linspace(1.0, x_max, 300)
    gamma_y = np.zeros_like(gamma_x)
    ax.plot(gamma_x, gamma_y, linewidth=3, label=r"removed ray $\Gamma$")

    # Mark (1,0) as excluded from Omega.
    ax.plot(1.0, 0.0, marker="o", markersize=8,
            markerfacecolor="white", markeredgewidth=1.5)
    ax.text(1.05, 0.08, r"$(1,0)\notin\Omega$", fontsize=10)

    # Finite-lifetime trajectory on y=0: choose x0=0.
    x0 = 0.0
    t_stop = 1.0 - x0
    t_finite = np.linspace(0.0, 0.995 * t_stop, 300)
    x_finite, y_finite = ex3_solution(t_finite, (x0, 0.0))
    ax.plot(
        x_finite,
        y_finite,
        linewidth=2.5,
        label=r"$y_0=0$: finite lifetime $m(0,0)=1$",
    )

    # Nearby initial data with y0 != 0 have infinite lifetime in Omega.
    nearby_y0 = [0.15, -0.15, 0.45, -0.45]
    t_global = np.linspace(0.0, x_max - x0, 500)
    for j, y0 in enumerate(nearby_y0):
        x_vals, y_vals = ex3_solution(t_global, (x0, y0))
        label = r"$y_0\ne0$: global in $\Omega$" if j == 0 else None
        ax.plot(x_vals, y_vals, linewidth=1.5, label=label)

    ax.axvline(1.0, linestyle=":", linewidth=1)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        r"Example 3: Domain Exit for $F(x,y)=(1,0)$ in "
        r"$\Omega=\mathbb{R}^2\setminus\Gamma$"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(
        "discontinuous_stopping_time_vector_field.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# ============================================================
# Example 4:
# Finite-time blow-up and a positive Koopman eigenfunction.
# System:
#   x' = y - x^2
#   y' = x - y
# ============================================================

def ex4_rhs(t, z):
    x, y = z
    return [y - x**2, x - y]


def ex4_stop_left(t, z):
    # Stop when x becomes very negative.
    x, y = z
    return x + 8.0


ex4_stop_left.terminal = True
ex4_stop_left.direction = -1


def ex4_stop_large_norm(t, z):
    x, y = z
    return 12.0 - np.sqrt(x**2 + y**2)


ex4_stop_large_norm.terminal = True
ex4_stop_large_norm.direction = -1


def plot_example4():
    x = np.linspace(-3, 3, 30)
    y = np.linspace(-3, 3, 30)
    X, Y = np.meshgrid(x, y)

    U = Y - X**2
    V = X - Y

    speed = np.sqrt(U**2 + V**2)
    U_plot = U / (1 + speed)
    V_plot = V / (1 + speed)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.streamplot(X, Y, U_plot, V_plot, density=1.2, linewidth=1)

    # Shade the negative quadrant.
    ax.axvspan(-3, 0, ymin=0, ymax=0.5, alpha=0.12)
    ax.axhspan(-3, 0, xmin=0, xmax=0.5, alpha=0.12)

    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)

    initial_points = [
        [-0.5, -0.5],
        [-1.0, -1.0],
        [-2.0, -1.0],
        [-1.5, -2.0],
        [0.5, 0.5],
        [1.0, 0.0],
    ]

    for z0 in initial_points:
        sol = solve_ivp(
            ex4_rhs,
            t_span=(0, 4),
            y0=z0,
            events=[ex4_stop_left, ex4_stop_large_norm],
            max_step=0.02,
            rtol=1e-8,
            atol=1e-10,
        )
        ax.plot(sol.y[0], sol.y[1], linewidth=1.5)
        ax.plot(z0[0], z0[1], marker="o", markersize=3)

    ax.text(-2.7, -2.7, "negative quadrant", fontsize=10)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"Example 4: Vector Field for $x'=y-x^2$, $y'=x-y$")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "koopman_blowup_vector_field.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    plot_example3()
    plot_example4()
