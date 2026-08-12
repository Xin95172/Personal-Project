from fastapi.testclient import TestClient

import pytest

from poker_solver.api.app import MultiwayActionRequest, MultiwayRootStrategyRequest, _multiway_action_from_ratio, create_app
from poker_solver.engine.river_game import Player, SizingPolicy, create_river_game
from poker_solver.engine.river_game import Action, ActionType
from poker_solver.engine.multiway_postflop_policy import MultiwayPostflopSizingPolicy
from poker_solver.engine.table import Position, advance_preflop_to_flop, apply_action, apply_multiway_postflop_action, create_8max_preflop
from poker_solver.solver_core.multiway_postflop_mccfr import MultiwayPostflopMCCFRTrainer
from poker_solver.solver_core.strategy_store import StrategyStore, store_river_root_strategy
from poker_solver.solver_core.river_mccfr import RiverMCCFRTrainer, WeightedRange


def app():
    trainer = RiverMCCFRTrainer(
        board=("As", "Kd", "Qh", "Jc", "2s"),
        oop_range=WeightedRange.from_cards((("Ts", "3d"),)),
        ip_range=WeightedRange.from_cards((("9h", "9c"),)),
        sizing_policy=SizingPolicy(bet_sizes=(0.5,), raise_sizes=(0.75,), max_re_raises=1),
        seed=2,
    )
    trainer.train(5)
    return create_app(trainer)


def test_health_reports_loaded_solver_metadata():
    response = TestClient(app()).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["solver_type"] == "river_heads_up"
    assert response.json()["iterations"] == 5


def test_river_strategy_query_returns_pot_ratio_actions():
    response = TestClient(app()).post(
        "/v1/river/strategy",
        json={
            "board": ["As", "Kd", "Qh", "Jc", "2s"],
            "hero_position": "oop",
            "hero_cards": ["Ts", "3d"],
            "pot_bb": 10,
            "effective_stack_bb": 95,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["pot"] == "10 BB"
    assert sum(item["probability"] for item in payload["actions"]) == 1.0
    assert any("pot" in item["action"] for item in payload["actions"])


def test_river_strategy_query_rejects_untrained_board():
    response = TestClient(app()).post(
        "/v1/river/strategy",
        json={
            "board": ["As", "Kd", "Qh", "Jc", "3s"],
            "hero_position": "oop",
            "hero_cards": ["Ts", "2d"],
            "pot_bb": 10,
            "effective_stack_bb": 95,
        },
    )
    assert response.status_code == 404


def test_river_strategy_query_rejects_duplicate_cards_before_lookup():
    response = TestClient(app()).post(
        "/v1/river/strategy",
        json={
            "board": ["As", "Kd", "Qh", "Jc", "2s"],
            "hero_position": "oop",
            "hero_cards": ["As", "3d"],
            "pot_bb": 10,
            "effective_stack_bb": 95,
        },
    )
    assert response.status_code == 422


def test_river_database_strategy_query_returns_pretrained_lookup(tmp_path):
    trainer = RiverMCCFRTrainer(
        board=("As", "Kd", "Qh", "Jc", "2s"),
        oop_range=WeightedRange.from_cards((("Ts", "3d"),)),
        ip_range=WeightedRange.from_cards((("9h", "9c"),)),
        sizing_policy=SizingPolicy(bet_sizes=(0.5,), raise_sizes=(0.75,), include_all_in=False, max_re_raises=0),
        seed=1,
    )
    trainer.train(3)
    store = StrategyStore(tmp_path / "strategies.sqlite3")
    state = create_river_game(trainer.board, ("Ts", "3d"), ("9h", "9c"))
    store_river_root_strategy(store, trainer, state, Player.OOP, range_profile_id="test-range")
    client = TestClient(create_app(trainer, strategy_store=store))
    response = client.post(
        "/v1/river/database-strategy",
        json={
            "board": ["As", "Kd", "Qh", "Jc", "2s"], "hero_position": "oop", "hero_cards": ["3d", "Ts"],
            "pot_bb": 10, "effective_stack_bb": 95, "range_profile_id": "test-range",
        },
    )
    assert response.status_code == 200
    assert response.json()["trained_iterations"] == 3
    assert sum(action["probability"] for action in response.json()["actions"]) == 1.0
    unified = client.post(
        "/v1/decision",
        json={
            "game_type": "river_heads_up", "street": "river", "player_count": 2,
            "hero_position": "oop", "hero_cards": ["Ts", "3d"], "board": ["As", "Kd", "Qh", "Jc", "2s"],
            "pot_bb": 10, "effective_stack_bb": 95, "range_profile_id": "test-range", "solver_version": "river-v1",
        },
    )
    assert unified.status_code == 200
    assert unified.json()["game_type"] == "river_heads_up"


def test_river_database_strategy_reports_missing_database_and_context():
    request = {"board": ["As", "Kd", "Qh", "Jc", "2s"], "hero_position": "oop", "hero_cards": ["Ts", "3d"], "pot_bb": 10, "effective_stack_bb": 95}
    assert TestClient(app()).post("/v1/river/database-strategy", json=request).status_code == 503


def test_river_checkpoint_rejects_multiway_endpoint():
    response = TestClient(app()).post(
        "/v1/multiway-postflop/root-strategy",
        json={"hero_position": "sb", "hero_cards": ["8c", "8d"]},
    )
    assert response.status_code == 404


def test_multiway_postflop_root_strategy_query_uses_pot_ratios():
    preflop = create_8max_preflop()
    for _ in range(6):
        preflop = apply_action(preflop, Action(ActionType.FOLD))
    preflop = apply_action(preflop, Action(ActionType.CALL))
    preflop = apply_action(preflop, Action(ActionType.CHECK))
    cards = (("2c", "2d"), ("3c", "3d"), ("4c", "4d"), ("5c", "5d"), ("6c", "6d"), ("7c", "7d"), ("8c", "8d"), ("9c", "9d"))
    trainer = MultiwayPostflopMCCFRTrainer(
        initial_state=advance_preflop_to_flop(preflop, ("As", "Kd", "Qh")),
        ranges={position: WeightedRange.from_cards((cards[index],)) for index, position in enumerate(Position)},
        sizing_policy=MultiwayPostflopSizingPolicy(bet_sizes=(0.5,), raise_sizes=(), include_all_in=False, max_re_raises=0),
    )
    trainer.train(5)
    health = TestClient(create_app(trainer)).get("/health")
    assert health.json()["solver_type"] == "multiway_postflop"
    wrong_endpoint = TestClient(create_app(trainer)).post(
        "/v1/river/strategy",
        json={"board": ["As", "Kd", "Qh", "Jc", "2s"], "hero_position": "oop", "hero_cards": ["Ts", "3d"], "pot_bb": 10, "effective_stack_bb": 95},
    )
    assert wrong_endpoint.status_code == 404
    response = TestClient(create_app(trainer)).post(
        "/v1/multiway-postflop/root-strategy",
        json={"hero_position": "sb", "hero_cards": ["8c", "8d"]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["street"] == "flop"
    assert sum(item["probability"] for item in payload["actions"]) == 1.0
    assert any("pot" in item["action"] for item in payload["actions"])
    after_bet = TestClient(create_app(trainer)).post(
        "/v1/multiway-postflop/root-strategy",
        json={
            "hero_position": "bb",
            "hero_cards": ["9c", "9d"],
            "actions": [{"kind": "bet", "pot_ratio": 0.5}],
        },
    )
    assert after_bet.status_code == 200
    assert any(item["action"] == "call" for item in after_bet.json()["actions"])


def test_multiway_api_request_validation_and_ratio_conversion():
    with pytest.raises(ValueError, match="hero cards"):
        MultiwayRootStrategyRequest(hero_position=Position.SB, hero_cards=("As", "As"))
    with pytest.raises(ValueError, match="pot_ratio"):
        MultiwayActionRequest(kind=ActionType.BET)
    with pytest.raises(ValueError, match="pot_ratio"):
        MultiwayActionRequest(kind=ActionType.CALL, pot_ratio=0.5)
    preflop = create_8max_preflop()
    for _ in range(6):
        preflop = apply_action(preflop, Action(ActionType.FOLD))
    preflop = apply_action(preflop, Action(ActionType.CALL))
    preflop = apply_action(preflop, Action(ActionType.CHECK))
    state = advance_preflop_to_flop(preflop, ("As", "Kd", "Qh"))
    assert _multiway_action_from_ratio(state, MultiwayActionRequest(kind=ActionType.CHECK)) == Action(ActionType.CHECK)
    assert _multiway_action_from_ratio(state, MultiwayActionRequest(kind=ActionType.BET, pot_ratio=0.5)) == Action(ActionType.BET, 100)
    state = apply_multiway_postflop_action(state, Action(ActionType.BET, 100))
    assert _multiway_action_from_ratio(state, MultiwayActionRequest(kind=ActionType.RAISE, pot_ratio=1.0)) == Action(ActionType.RAISE, 500)
