import matplotlib.pyplot as plt


def riccati_stopping_time(s, y0, h=1e-3, M=100, max_steps=1_000_000):
    """
    Approximate the maximal forward lifetime for

        y'(t) = t^2 + y(t)^2,   y(s) = y0

    by applying explicit Euler to the lifted autonomous system

        x' = 1,
        w' = x^2 + w^2,
        x(0) = s,
        w(0) = y0.

    The computation stops when |w_n| >= M.

    Returns the elapsed forward lifetime n*h, rather than the
    physical threshold-crossing time s + n*h.
    """

    x = float(s)
    w = float(y0)

    for n in range(max_steps + 1):
        if abs(w) >= M:
            stopping_lifetime = n * h
            return stopping_lifetime, n, x, w

        x_new = x + h
        w_new = w + h * (x**2 + w**2)

        x = x_new
        w = w_new

    return None, max_steps, x, w


def riccati_trajectory(s, y0, h=1e-3, M=100, max_steps=1_000_000):
    """
    Compute the Euler trajectory for the lifted Riccati system until the
    threshold |w_n| >= M is reached or max_steps is exceeded.

    The returned stopping_lifetime is the elapsed forward time n*h.
    The returned physical_stopping_time is s + n*h.
    """

    x = float(s)
    w = float(y0)

    # These times are physical times t = s + tau.
    times = [s]
    x_vals = [x]
    w_vals = [w]

    for n in range(max_steps):
        if abs(w) >= M:
            stopping_lifetime = n * h
            physical_stopping_time = s + stopping_lifetime

            return (
                times,
                x_vals,
                w_vals,
                stopping_lifetime,
                physical_stopping_time,
                n,
            )

        x_new = x + h
        w_new = w + h * (x**2 + w**2)

        x = x_new
        w = w_new

        times.append(s + (n + 1) * h)
        x_vals.append(x)
        w_vals.append(w)

    return (
        times,
        x_vals,
        w_vals,
        None,
        None,
        max_steps,
    )


def plot_riccati_trajectory(
    s=0.0,
    y0=1.0,
    h=1e-3,
    M=100,
    max_steps=1_000_000,
    save_path="riccati_blowup_plot.png",
):
    """
    Plot w_n versus physical time for one threshold M and mark the
    approximate threshold-crossing time.
    """

    (
        times,
        x_vals,
        w_vals,
        stopping_lifetime,
        physical_stopping_time,
        stop_index,
    ) = riccati_trajectory(
        s=s,
        y0=y0,
        h=h,
        M=M,
        max_steps=max_steps,
    )

    plt.figure(figsize=(10, 6))
    plt.plot(times, w_vals, label="Euler approximation of $w_n$")
    plt.axhline(M, linestyle="--", label=f"Threshold $M={M}$")

    if physical_stopping_time is not None:
        plt.axvline(
            physical_stopping_time,
            linestyle=":",
            label=(
                f"Approx. stopping lifetime = "
                f"{stopping_lifetime:.3f}"
            ),
        )

        plt.plot(
            times[-1],
            w_vals[-1],
            marker="o",
            linestyle="None",
            label="Stopping point",
        )

    plt.xlabel("Physical time $t$")
    plt.ylabel("$w_n$")
    plt.title(
        "Explicit Euler Approximation for the Lifted Riccati System"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved single-threshold plot as: {save_path}")

    return (
        stopping_lifetime,
        physical_stopping_time,
        stop_index,
        times[-1],
        w_vals[-1],
    )


def plot_riccati_thresholds(
    s=0.0,
    y0=1.0,
    h=1e-3,
    thresholds=None,
    max_steps=1_000_000,
    save_path="riccati_threshold_comparison.png",
):
    """
    Plot w_n versus physical time for several threshold values M.

    Each trajectory is stopped when |w_n| first exceeds its threshold.
    """

    if thresholds is None:
        thresholds = [50, 100, 500]

    plt.figure(figsize=(10, 6))

    for M in thresholds:
        (
            times,
            x_vals,
            w_vals,
            stopping_lifetime,
            physical_stopping_time,
            stop_index,
        ) = riccati_trajectory(
            s=s,
            y0=y0,
            h=h,
            M=M,
            max_steps=max_steps,
        )

        plt.plot(times, w_vals, label=f"$M={M}$")
        plt.axhline(M, linestyle="--")

        if physical_stopping_time is not None:
            plt.axvline(
                physical_stopping_time,
                linestyle=":",
            )

            plt.plot(
                times[-1],
                w_vals[-1],
                marker="o",
                linestyle="None",
            )

    plt.xlabel("Physical time $t$")
    plt.ylabel("$w_n$")
    plt.title(
        "Explicit Euler Approximation for the Lifted Riccati System"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved multi-threshold plot as: {save_path}")


def run_threshold_experiment():
    """
    Run the Riccati stopping-time approximation for several thresholds
    and print the resulting forward-lifetime estimates.
    """

    s = 0.0
    y0 = 1.0
    h = 1e-3
    thresholds = [50, 100, 500]

    print("Riccati Stopping-Time Approximation")
    print("-----------------------------------")
    print("Equation: y'(t) = t^2 + y(t)^2")
    print(f"Initial condition: y({s}) = {y0}")
    print(f"Step size: h = {h}")
    print()
    print("Comparison estimate: m(0,1) <= 1")
    print()

    for M in thresholds:
        stopping_lifetime, n, x_final, w_final = (
            riccati_stopping_time(
                s=s,
                y0=y0,
                h=h,
                M=M,
            )
        )

        if stopping_lifetime is not None:
            physical_stopping_time = s + stopping_lifetime
        else:
            physical_stopping_time = None

        print(f"Threshold M = {M}")
        print(
            f"  Approximate stopping lifetime: "
            f"{stopping_lifetime}"
        )
        print(
            f"  Physical threshold-crossing time: "
            f"{physical_stopping_time}"
        )
        print(f"  Steps taken: {n}")
        print(f"  Final x value: {x_final}")
        print(f"  Final w value: {w_final}")
        print()


if __name__ == "__main__":
    # Print stopping-time approximations for the thresholds used
    # in the thesis.
    run_threshold_experiment()

    # Generate a single-threshold representative plot.
    plot_riccati_trajectory(
        s=0.0,
        y0=1.0,
        h=1e-3,
        M=100,
        save_path="riccati_blowup_plot.png",
    )

    # Generate a multi-threshold comparison plot.
    plot_riccati_thresholds(
        s=0.0,
        y0=1.0,
        h=1e-3,
        thresholds=[50, 100, 500],
        save_path="riccati_threshold_comparison.png",
    )