import numpy as np

from src.assembly.solver import _grid_shape
from src.assembly.solver import _search_grid


def piece_edges():
    return {
        "top": {"angle_degrees": 0},
        "right": {"angle_degrees": 90},
        "bottom": {"angle_degrees": 180},
        "left": {"angle_degrees": -90},
    }


def test_grid_shape_uses_piece_proportions():
    images = {
        f"P{number}": np.zeros((170, 102, 4), dtype=np.uint8)
        for number in range(15)
    }

    assert _grid_shape(images) == (3, 5)


def test_search_grid_uses_global_neighbors():
    pieces = ["A", "B", "C", "D"]
    scores = {
        ("A", "B", "horizontal"): 1.0,
        ("C", "D", "horizontal"): 1.0,
        ("A", "C", "vertical"): 1.0,
        ("B", "D", "vertical"): 1.0,
    }
    edges = {piece: piece_edges() for piece in pieces}

    order, score = _search_grid(
        pieces,
        rows=2,
        columns=2,
        scores=scores,
        piece_edges=edges,
        beam_width=100,
    )

    assert order == ["A", "B", "C", "D"]
    assert score > 4
