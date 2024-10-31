import jax
import jax.numpy as jnp
import numpy as np
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


def generate_initial_grid(
        size: int,
        proportion_alive: float = 0.5,
        seed : int = 0
) -> Int[Array, ' size size']:
    """Generates a random grid of zeros and ones (int32)"""
    key = jax.random.key(seed)
    return jnp.where(
        jax.random.bernoulli(key, proportion_alive, (size, size)), 1, 0)


def unoptimized_gameoflife(grid: Int[Array, '...'], n_iter: int) -> Int[Array, '...']:
    """Runs the Game of Life for n_iter iterations using Python loops."""
    grid = np.array(grid)
    for _ in range(n_iter):
        new_grid = grid.copy()  # Create a copy to store updates
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                neighbor_count = 0
                for i_n in range(i - 1, i + 2):
                    for j_n in range(j - 1, j + 2):
                        if (0 <= i_n < len(grid)) and (0 <= j_n < len(grid[0])):
                            neighbor_count += grid[i_n][j_n]
                neighbor_count -= grid[i][j]  # Exclude the cell itself
                # Apply the Game of Life rules
                if grid[i][j] == 1:
                    if neighbor_count < 2 or neighbor_count > 3:
                        new_grid[i][j] = 0
                else:
                    if neighbor_count == 3:
                        new_grid[i][j] = 1
        grid = new_grid  # Update the grid after all cells have been processed
    return jnp.array(grid)


def optimized_gameoflife(grid: Int[Array, '...'], n_iter: int) -> Int[Array, '...']:
    """Runs the Game of Life for n_iter iterations using JAX functions."""
    for _ in range(n_iter):
        grid = next_turn(grid)
    return grid


@jax.jit
def compiled_gameoflife(grid: Int[Array, '...'], n_iter: int) -> Int[Array, '...']:
    """Compiles the optimized_gameoflife function."""
    return jax.lax.fori_loop(0, n_iter, lambda i, grid: next_turn(grid), grid)
