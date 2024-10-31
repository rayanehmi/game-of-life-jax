import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Int
import time
import matplotlib.pyplot as plt


def augment_grid(grid: Int[Array, '...']) -> Int[Array, '...']:
    """Creates a new array with the surrounding zeros"""
    ret = jnp.pad(grid, ((1, 1), (1, 1)))
    return ret


def compute_neighbors(grid: Int[Array, '...']) -> Int[Array, '...']:
    """Returns an array with the sum of neighbors for each cell."""
    augmented_grid = augment_grid(grid)
    left = jnp.roll(augmented_grid, -1)
    right = jnp.roll(augmented_grid, 1)
    down = jnp.roll(augmented_grid, shift=[1, 0], axis=[0, 1])
    up = jnp.roll(augmented_grid, shift=[-1, 0], axis=[0, 1])
    left_up = jnp.roll(augmented_grid, shift=[-1, -1], axis=[0, 1])
    right_up = jnp.roll(augmented_grid, shift=[-1, 1], axis=[0, 1])
    down_left = jnp.roll(augmented_grid, shift=[1, -1], axis=[0, 1])
    down_right = jnp.roll(augmented_grid, shift=[1, 1], axis=[0, 1])
    summation = up + down + left + right + left_up + right_up + down_left + down_right
    cropped_sum = summation[1: -1, 1:-1]
    return cropped_sum


def next_turn(grid: Int[Array, '...']) -> Int[Array, '...']:
    """Returns a non-augmented evolved grid"""
    neighbor_grid = compute_neighbors(grid)
    first_condition = (grid == 1) & ((neighbor_grid == 2) | (neighbor_grid == 3))
    second_condition = (grid == 0) & (neighbor_grid == 3)
    next_grid = jnp.where(first_condition | second_condition, 1, 0)
    return next_grid


def benchmark(init_grid: Int[Array, '...']) -> None:
    """Benchmarks the performance of the next_turn function"""

    num_iter_list = [1, 500, 1000, 5000, 10000]
    compiled_times = []
    not_compiled_times = []

    # compiled
    for num_iter in num_iter_list:
        start_time = time.time()
        _ = jax.lax.fori_loop(
            lower = 0,
            upper = num_iter,
            body_fun = lambda i, x: next_turn(x),
            init_val = init_grid
        )
        end_time = time.time()
        compiled_times.append(end_time - start_time)

    # not compiled
    for num_iter in num_iter_list:
        start_time = time.time()
        for i in range(num_iter):
            init_grid = next_turn(init_grid)
        end_time = time.time()
        not_compiled_times.append(end_time - start_time)

    plt.scatter(num_iter_list, compiled_times, label="compiled")
    plt.scatter(num_iter_list, not_compiled_times, label="not compiled")
    plt.xlabel("Number of iterations")
    plt.ylabel("Time taken (seconds)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    print("available devices :", jax.devices())

    blinker = jnp.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])

    benchmark(blinker)




