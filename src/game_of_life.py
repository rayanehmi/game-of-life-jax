"""
An implementation of Conway's Game of Life using the JAX library.

The core mechanics are:
1. `augment_grid`: Adds padding of zeros around the grid to handle edge cases
2. `compute_neighbors`: Uses JAX's roll operations to efficiently calculate neighbor sums by shifting the grid in all 8 directions
3. `next_turn`: Applies Conway's rules using boolean operations:
   - Cells survive if they have 2-3 neighbors
   - Dead cells with exactly 3 neighbors become alive
   - All other cells die
The implementation is optimized for performance using JAX's vectorized operations and supports both CPU and GPU execution.

author: rayanehmi@github
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Int


def augment_grid(grid: Int[Array, "..."]) -> Int[Array, "..."]:
    """Creates a new array with zeros padding on each side."""
    ret = jnp.pad(grid, ((1, 1), (1, 1)))
    return ret


def compute_neighbors(grid: Int[Array, "..."]) -> Int[Array, "..."]:
    """Returns an array where each cell is the number of non-zero neightbors of that cell in "grid"."""
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
    cropped_sum = summation[1:-1, 1:-1]
    return cropped_sum


def next_turn(grid: Int[Array, "..."]) -> Int[Array, "..."]:
    """Applies one turn of the Game of Life rules to the grid and returns the new grid."""
    neighbor_grid = compute_neighbors(grid)
    first_condition = (grid == 1) & ((neighbor_grid == 2) | (neighbor_grid == 3))
    second_condition = (grid == 0) & (neighbor_grid == 3)
    next_grid = jnp.where(first_condition | second_condition, 1, 0)
    return next_grid


def generate_initial_grid(
    size: int, proportion_alive: float = 0.5, seed: int = 0
) -> Int[Array, "..."]:
    """Generates a (size, size) grid randomly filled with zeros and ones (int32)"""
    key = jax.random.key(seed)
    return jnp.where(jax.random.bernoulli(key, proportion_alive, (size, size)), 1, 0)


def unoptimized_gameoflife(grid: Int[Array, "..."], n_iter: int) -> Int[Array, "..."]:
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


def optimized_gameoflife(grid: Int[Array, "..."], n_iter: int) -> Int[Array, "..."]:
    """Runs the Game of Life for n_iter iterations using JAX functions in a python loop."""
    for _ in range(n_iter):
        grid = next_turn(grid)
    return grid


@jax.jit
def compiled_gameoflife(grid: Int[Array, "..."], n_iter: int) -> Int[Array, "..."]:
    """Compiles the optimized_gameoflife function."""
    return jax.lax.fori_loop(0, n_iter, lambda i, grid: next_turn(grid), grid)
