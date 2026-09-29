import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ============================================================
# Example 3:
# A Discontinuous Stopping-Time Function
# System:
#   x' = x^2(1 - y^2)
#   y' = x y
# ============================================================

def ex3_rhs(t, z):
    x, y = z
    return [x**2 * (1 - y**2), x * y]


def ex3_stop_large_x(t, z):
    # Stop if x becomes too large (to avoid blow-up wrecking the plot)
    x, y = z
    return 8.0 - x

ex3_stop_large_x.terminal = True
ex3_stop_large_x.direction = -1


def plot_example3():
    # Grid for vector field
    x = np.linspace(0.0, 2.0, 30)
    y = np.linspace(-1.5, 1.5, 30)
    X, Y = np.meshgrid(x, y)

    U = X**2 * (1 - Y**2)
    V = X * Y

    speed = np.sqrt(U**2 + V**2)
    # Avoid huge arrows dominating appearance
    U_plot = U / (1 + speed)
    V_plot = V / (1 + speed)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Stream plot of the vector field
    ax.streamplot(X, Y, U_plot, V_plot, density=1.2, linewidth=1)

    # Plot the invariant line y = 0
    ax.axhline(0, linestyle='--', linewidth=1)

    # Mark the key point (1,0)
    ax.plot(1, 0, marker='o')
    ax.text(1.03, 0.05, "(1,0)", fontsize=10)

    # Exact trajectory on y = 0 with x(0)=1:
    # x(t) = 1 / (1 - t), y(t)=0, which blows up at t=1
    t_exact = np.linspace(0, 0.87, 300)
    x_exact = 1.0 / (1.0 - t_exact)
    y_exact = np.zeros_like(t_exact)

    # Clip to plotting window
    mask = x_exact <= 2.0
    ax.plot(x_exact[mask], y_exact[mask], linewidth=2, label="trajectory from (1,0)")

    # Nearby trajectories with y0 ≠ 0
    initial_points = [
        [1.0, 0.2],
        [1.0, -0.2],
        [1.0, 0.5],
        [1.0, -0.5]
    ]

    for z0 in initial_points:
        sol = solve_ivp(
            ex3_rhs,
            t_span=(0, 4),
            y0=z0,
            events=ex3_stop_large_x,
            max_step=0.02,
            rtol=1e-8,
            atol=1e-10
        )
        ax.plot(sol.y[0], sol.y[1], linewidth=1.5)

    ax.set_xlim(0, 2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Example 3: Vector Field for $F(x,y)=(x^2(1-y^2),xy)$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("discontinuous_stopping_time_vector_field.png", dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# Example 4:
# Finite-Time Blow-up and a Positive Koopman Eigenfunction
# System:
#   x' = y - x^2
#   y' = x - y
# ============================================================

def ex4_rhs(t, z):
    x, y = z
    return [y - x**2, x - y]


def ex4_stop_left(t, z):
    # Stop when x becomes very negative
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
    # Grid for vector field
    x = np.linspace(-3, 3, 30)
    y = np.linspace(-3, 3, 30)
    X, Y = np.meshgrid(x, y)

    U = Y - X**2
    V = X - Y

    speed = np.sqrt(U**2 + V**2)
    U_plot = U / (1 + speed)
    V_plot = V / (1 + speed)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Stream plot of the vector field
    ax.streamplot(X, Y, U_plot, V_plot, density=1.2, linewidth=1)

    # Shade the negative quadrant
    ax.axvspan(-3, 0, ymin=0, ymax=0.5, alpha=0.12)
    ax.axhspan(-3, 0, xmin=0, xmax=0.5, alpha=0.12)

    # Boundary axes
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)

    # Sample trajectories, especially in the negative quadrant
    initial_points = [
        [-0.5, -0.5],
        [-1.0, -1.0],
        [-2.0, -1.0],
        [-1.5, -2.0],
        [0.5, 0.5],
        [1.0, 0.0]
    ]

    for z0 in initial_points:
        sol = solve_ivp(
            ex4_rhs,
            t_span=(0, 4),
            y0=z0,
            events=[ex4_stop_left, ex4_stop_large_norm],
            max_step=0.02,
            rtol=1e-8,
            atol=1e-10
        )
        ax.plot(sol.y[0], sol.y[1], linewidth=1.5)
        ax.plot(z0[0], z0[1], marker='o', markersize=3)

    ax.text(-2.7, -2.7, "negative quadrant", fontsize=10)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Example 4: Vector Field for $x'=y-x^2$, $y'=x-y$")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("koopman_blowup_vector_field.png", dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# Run both plots
# ============================================================

if __name__ == "__main__":
    plot_example3()
    plot_example4()