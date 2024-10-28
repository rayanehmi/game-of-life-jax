from src.plotting import plot_grid
import jax.numpy as jnp
import os

def test_plot_grid(tmp_path: str):

    grid = jnp.array([[0, 1], [2, 0]])

    grid_returned = plot_grid(
        grid=grid,
        show=False,
        save_path=tmp_path,
        name="grid_test.png",
    )

    target = jnp.array([[False, True], [True, False]])
    assert jnp.all(grid_returned == target), "Grid not converted correctly."

    assert os.path.exists(
        os.path.join(tmp_path, "grid_test.png")
    ), "Grid plot not saved."



