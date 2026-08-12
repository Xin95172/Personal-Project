import pytest

from poker_solver.engine.money import bb_to_units as bb
from poker_solver.engine.river_game import Action, ActionType, Player, SizingPolicy, abstract_actions, apply_action, create_river_game
from poker_solver.solver_core.river_mccfr import Combo, RiverMCCFRTrainer, TurnRiverMCCFRTrainer, WeightedRange


BOARD = ("As", "Kd", "Qh", "Jc", "2s")


def make_trainer(seed=7):
    return RiverMCCFRTrainer(
        board=BOARD,
        oop_range=WeightedRange.from_cards((("Ts", "3d"), ("9h", "9c"))),
        ip_range=WeightedRange.from_cards((("Tc", "4d"), ("8h", "8c"))),
        sizing_policy=SizingPolicy(bet_sizes=(0.5,), raise_sizes=(0.75,), max_re_raises=1),
        seed=seed,
    )


def test_combo_and_range_validate_their_input():
    with pytest.raises(ValueError, match="distinct"):
        Combo(("As", "As"))
    with pytest.raises(ValueError, match="positive"):
        Combo(("As", "Kd"), weight=0)
    with pytest.raises(ValueError, match="at least one"):
        WeightedRange(())
    with pytest.raises(ValueError, match="duplicate"):
        WeightedRange((Combo(("As", "Kd")), Combo(("As", "Kd"))))


def test_trainer_rejects_ranges_with_no_possible_deal():
    trainer = RiverMCCFRTrainer(
        board=BOARD,
        oop_range=WeightedRange.from_cards((("Ts", "3d"),)),
        ip_range=WeightedRange.from_cards((("Ts", "4d"),)),
    )
    with pytest.raises(ValueError, match="cannot produce"):
        trainer.train(1)


def test_mccfr_training_creates_normalized_average_strategies():
    trainer = make_trainer()
    stats = trainer.train(20)
    assert stats.iterations == 20
    assert stats.total_iterations == 20
    assert stats.infosets > 0
    assert stats.iterations_per_second > 0
    assert stats.mean_positive_regret >= 0

    for node in trainer.infosets.values():
        strategy = node.average_strategy()
        assert set(strategy) == set(node.actions)
        assert sum(strategy.values()) == pytest.approx(1.0)
        assert node.visit_count > 0


def test_strategy_for_returns_the_trained_root_infoset():
    trainer = make_trainer()
    trainer.train(10)
    state = create_river_game(BOARD, ("Ts", "3d"), ("Tc", "4d"))
    strategy = trainer.strategy_for(state, Player.OOP)
    assert set(strategy) == set(abstract_actions(state, trainer.sizing_policy))
    assert sum(strategy.values()) == pytest.approx(1.0)


def test_strategy_for_rejects_an_unvisited_infoset():
    trainer = make_trainer()
    state = create_river_game(BOARD, ("Ts", "3d"), ("Tc", "4d"))
    with pytest.raises(KeyError, match="not been visited"):
        trainer.strategy_for(state, Player.OOP)


def test_policy_caps_solver_re_raises_but_does_not_change_engine_rules():
    policy = SizingPolicy(bet_sizes=(0.5,), raise_sizes=(0.75,), max_re_raises=1)
    state = create_river_game(BOARD, ("Ts", "3d"), ("Tc", "4d"))
    state = apply_action(state, Action(ActionType.BET, bb(5)))
    state = apply_action(state, Action(ActionType.RAISE, bb(20)))
    # The engine can still validate a re-raise, but this policy does not expose it to the solver.
    assert Action(ActionType.RAISE, bb(50)) not in abstract_actions(state, policy)
    assert Action(ActionType.CALL) in abstract_actions(state, policy)


def test_trainer_requires_positive_iteration_count():
    with pytest.raises(ValueError, match="positive"):
        make_trainer().train(0)


def test_turn_river_trainer_samples_river_cards_during_training():
    trainer = TurnRiverMCCFRTrainer(
        turn_board=("As", "Kd", "Qh", "Jc"),
        oop_range=WeightedRange.from_cards((("Ts", "3d"),)),
        ip_range=WeightedRange.from_cards((("9h", "9c"),)),
        sizing_policy=SizingPolicy(bet_sizes=(0.5,), raise_sizes=(0.75,), max_re_raises=1),
        seed=4,
    )
    stats = trainer.train(20)
    boards = {key[2] for key in trainer.infosets}
    assert stats.infosets > 0
    assert all(len(board) == 5 for board in boards)
    assert len(boards) > 1


def test_turn_river_trainer_requires_four_board_cards():
    with pytest.raises(ValueError, match="four"):
        TurnRiverMCCFRTrainer(
            turn_board=("As", "Kd", "Qh"),
            oop_range=WeightedRange.from_cards((("Ts", "3d"),)),
            ip_range=WeightedRange.from_cards((("9h", "9c"),)),
        )
