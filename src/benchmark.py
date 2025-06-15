import jax
import matplotlib.pyplot as plt

from game_of_life import (
    generate_initial_grid,
    unoptimized_gameoflife,
    optimized_gameoflife,
    compiled_gameoflife,
)
import timeit


def full_benchmark() -> None:
    """Benchmarks the performance of the next_turn function.
    Compares python loops, JAX numpy arrays and compiled JAX loops."""

    sizes = [10, 100, 250, 500, 1000]
    times_unoptimized = []
    times_optimized = []
    times_compiled = []

    for size in sizes:
        print(f"Running benchmark for size {size}")
        grid = generate_initial_grid(size)
        grid_unoptimized = grid.copy()
        grid_optimized = grid.copy()
        grid_compiled = grid.copy()

        if size < 1000:
            times_unoptimized.append(
                timeit.timeit(
                    lambda: unoptimized_gameoflife(grid_unoptimized, 100), number=1
                )
            )
        if size < 5000:
            times_optimized.append(
                timeit.timeit(
                    lambda: optimized_gameoflife(grid_optimized, 100), number=1
                )
            )
        times_compiled.append(
            timeit.timeit(lambda: compiled_gameoflife(grid_compiled, 100), number=1)
        )

    plt.plot(sizes[: len(times_unoptimized)], times_unoptimized, label="Unoptimized")
    plt.plot(sizes[: len(times_optimized)], times_optimized, label="Optimized")
    plt.plot(sizes, times_compiled, label="Compiled")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Grid size")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.show()


def benchmark_gpu_only() -> None:
    """Benchmarks the performance of the next_turn function"""

    print(f"Current platform: {jax.default_backend()}")

    sizes = [10, 100, 250, 500, 1000]
    times_compiled = []
    for size in sizes:
        print(f"Running benchmark for size {size}")
        grid_compiled = generate_initial_grid(size)
        times_compiled.append(
            timeit.timeit(lambda: compiled_gameoflife(grid_compiled, 5000), number=1)
        )

    plt.plot(sizes, times_compiled, label="Compiled")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Grid size")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    full_benchmark()
