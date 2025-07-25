from src.game_of_life import (
    augment_grid,
    compute_neighbors,
    next_turn,
    generate_initial_grid,
    unoptimized_gameoflife,
    optimized_gameoflife,
    compiled_gameoflife,
)
import jax.numpy as jnp


def test_augment_grid():
    test_array = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    augmented_array = augment_grid(test_array)
    assert jnp.all(
        augmented_array
        == jnp.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ]
        )
    ), "Padding did not work"


def test_compute_neighbors():
    test_array = jnp.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    target = jnp.array(
        [
            [2, 3, 2],
            [1, 2, 1],
            [2, 3, 2],
        ]
    )
    neighbors = compute_neighbors(test_array)
    assert jnp.all(neighbors == target)


def test_next_turn():
    test_array = jnp.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    next_array = next_turn(test_array)
    target = jnp.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    assert jnp.all(next_array == target)


def test_generate_initial_grid_with_full_proportion_alive():
    grid = generate_initial_grid(3, proportion_alive=1.0)
    assert jnp.all(grid == 1), "Grid generation failed for full proportion alive"


def test_generate_initial_grid_with_no_proportion_alive():
    grid = generate_initial_grid(3, proportion_alive=0.0)
    assert jnp.all(grid == 0), "Grid generation failed for no proportion alive"


def test_all_gameoflife():
    glider = jnp.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]])
    for k in range(3):
        glider = augment_grid(glider)

    for i in range(3):
        glider_optimized = glider
        glider_unoptimized = glider
        glider_compiled = glider

        glider_optimized = optimized_gameoflife(glider_optimized, i)
        glider_unoptimized = unoptimized_gameoflife(glider_unoptimized, i)
        glider_compiled = compiled_gameoflife(glider_compiled, i)

        assert jnp.all(glider_optimized == glider_unoptimized), (
            "Optimized and unoptimized not equal"
        )
        assert jnp.all(glider_optimized == glider_compiled), (
            "Optimized and compiled not equal"
        )


def test_unoptimized_gameoflife():
    # Test with a blinker pattern (oscillator)
    blinker = jnp.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])

    # After one iteration, the blinker should be vertical
    result_1 = unoptimized_gameoflife(blinker, 1)
    expected_1 = jnp.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    assert jnp.all(result_1 == expected_1), "Blinker pattern failed after 1 iteration"

    # After two iterations, it should be back to horizontal
    result_2 = unoptimized_gameoflife(blinker, 2)
    assert jnp.all(result_2 == blinker), "Blinker pattern failed after 2 iterations"

    # Test with a block (still life)
    block = jnp.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])

    # Block should remain unchanged after any number of iterations
    result_block = unoptimized_gameoflife(block, 5)
    assert jnp.all(result_block == block), "Block pattern failed to remain stable"
