import pytest

from poker_solver.engine.river_game import create_river_game


@pytest.fixture
def winning_oop_game():
    """OOP has Broadway; IP has one pair."""
    return create_river_game(
        ("As", "Kd", "Qh", "Jc", "2s"),
        ("Ts", "3d"),
        ("9h", "9c"),
    )


@pytest.fixture
def split_pot_game():
    """The board plays for both players."""
    return create_river_game(
        ("Ah", "Kh", "Qh", "Jh", "Th"),
        ("2c", "3d"),
        ("4c", "5d"),
    )
