import pytest

from poker_solver.engine.postflop_game import create_turn_game
from poker_solver.engine.river_game import Player, SizingPolicy
from poker_solver.solver_core.river_mccfr import WeightedRange
from poker_solver.solver_core.turn_mccfr import FlopMCCFRTrainer, TurnMCCFRTrainer


def trainer():
    return TurnMCCFRTrainer(
        turn_board=("As", "Kd", "Qh", "Jc"),
        oop_range=WeightedRange.from_cards((("Ts", "3d"),)),
        ip_range=WeightedRange.from_cards((("9h", "9c"),)),
        sizing_policy=SizingPolicy(bet_sizes=(0.5,), raise_sizes=(0.75,), max_re_raises=1),
        seed=3,
    )


def test_turn_mccfr_trains_across_turn_and_river_infosets():
    value = trainer()
    stats = value.train(10)
    streets = {key[0] for key in value.infosets}
    assert stats.iterations == 10
    assert {"turn", "river"}.issubset(streets)
    for node in value.infosets.values():
        assert sum(node.average_strategy().values()) == pytest.approx(1.0)


def test_turn_mccfr_exposes_root_strategy():
    value = trainer()
    value.train(5)
    state = create_turn_game(("As", "Kd", "Qh", "Jc"), ("Ts", "3d"), ("9h", "9c"))
    strategy = value.strategy_for(state, Player.OOP)
    assert sum(strategy.values()) == pytest.approx(1.0)


def test_turn_mccfr_rejects_invalid_iteration_count():
    with pytest.raises(ValueError, match="positive"):
        trainer().train(0)


def test_flop_mccfr_trains_across_all_three_streets():
    value = FlopMCCFRTrainer(
        flop_board=("As", "Kd", "Qh"),
        oop_range=WeightedRange.from_cards((("Ts", "3d"),)),
        ip_range=WeightedRange.from_cards((("9h", "9c"),)),
        sizing_policy=SizingPolicy(bet_sizes=(0.5,), raise_sizes=(0.75,), max_re_raises=1),
        seed=8,
    )
    value.train(10)
    assert {"flop", "turn", "river"}.issubset({key[0] for key in value.infosets})


def test_flop_mccfr_requires_three_board_cards():
    with pytest.raises(ValueError, match="three"):
        FlopMCCFRTrainer(
            flop_board=("As", "Kd"),
            oop_range=WeightedRange.from_cards((("Ts", "3d"),)),
            ip_range=WeightedRange.from_cards((("9h", "9c"),)),
        )
