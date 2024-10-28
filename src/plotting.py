import matplotlib.pyplot as plt
from jax import Array
import os
from jaxtyping import Bool, Int


def plot_grid(
        grid: Int[Array, '...'],
        show: bool = True,
        save_path: str = None,
        name: str = "grid.png"
) -> Bool[Array, '...']:
    """
    Plots the grid.
    The grid is a 2D jax numpy array of integers. The function needs first to convert the grid to a boolean grid where
    values higher than 1 are converted to 1.

    :param grid: 2D jax numpy array of integers
    :param show: whether to show the plot
    :param save_path: path to save the plot
    :param name: name of the plot

    :return: None
    """
    # Part 1 : convert values higher than 1 to 1
    grid_bool = grid > 0

    # Part 2 : plot the grid
    plt.imshow(grid_bool, cmap='gray')
    plt.gca().invert_yaxis()
    plt.xticks(range(grid_bool.shape[1]))
    plt.yticks(range(grid_bool.shape[0]))

    if show:
        plt.show()

    if save_path and name:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, name))

    plt.close()

    return grid_bool
