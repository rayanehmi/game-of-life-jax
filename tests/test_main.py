from src.main import augment_grid, compute_neighbors, next_turn
import jax.numpy as jnp


def test_augment_grid():
    test_array = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    augmented_array = augment_grid(test_array)
    assert jnp.all(augmented_array == jnp.array(
        [[0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0]]
    )), "Padding did not work"


def test_compute_neighbors():
    test_array = jnp.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    target = jnp.array([
        [2, 3, 2],
        [1, 2, 1],
        [2, 3, 2],
    ])
    neighbors = compute_neighbors(test_array)
    assert jnp.all(neighbors == target)


def test_next_turn():
    test_array = jnp.array([
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0]
    ])
    next_array = next_turn(test_array)
    target = jnp.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ])
    assert jnp.all(next_array == target)
