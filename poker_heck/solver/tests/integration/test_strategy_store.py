from poker_solver.engine.multiway_postflop_policy import MultiwayPostflopSizingPolicy
from argparse import ArgumentParser
import json
from pathlib import Path

import pytest
from poker_solver.cli.build_strategy_db import _run_job
from poker_solver.generators.heads_up import build_jobs as build_heads_up_jobs
from poker_solver.engine.preflop_policy import PreflopSizingPolicy
from poker_solver.engine.river_game import Action, ActionType, Player, SizingPolicy, create_river_game
from poker_solver.engine.postflop_game import create_flop_game, create_turn_game
from poker_solver.engine.table import Position, advance_preflop_to_flop, apply_action, create_8max_preflop
from poker_solver.solver_core.multiway_postflop_mccfr import MultiwayPostflopMCCFRTrainer
from poker_solver.solver_core.preflop_mccfr import MultiwayPreflopMCCFRTrainer
from poker_solver.solver_core.river_analysis import analyze_river_profile
from poker_solver.solver_core.river_mccfr import RiverMCCFRTrainer, WeightedRange
from poker_solver.solver_core.turn_mccfr import FlopMCCFRTrainer, TurnMCCFRTrainer
from poker_solver.solver_core.strategy_store import (
    StrategyStore,
    StoredAction,
    StoredStrategy,
    strategy_key,
    river_strategy_key,
    store_multiway_postflop_root_strategy,
    store_preflop_root_strategy,
    store_river_root_strategy,
    export_river_tree,
    export_preflop_infosets,
    export_multiway_postflop_infosets,
    export_multiway_postflop_root_infosets,
    export_heads_up_postflop_infosets,
    store_heads_up_postflop_root_strategy,
)
from poker_solver.solver_core.strategy_store import StrategyContext


def _trainer() -> RiverMCCFRTrainer:
    trainer = RiverMCCFRTrainer(
        board=("As", "Kd", "Qh", "Jc", "2s"),
        oop_range=WeightedRange.from_cards((("Ts", "3d"),)),
        ip_range=WeightedRange.from_cards((("9h", "9c"),)),
        sizing_policy=SizingPolicy(bet_sizes=(0.5,), raise_sizes=(0.75,), include_all_in=False, max_re_raises=0),
        seed=4,
    )
    trainer.train(5)
    return trainer


def test_river_strategy_store_round_trip_and_card_order_independent_lookup(tmp_path):
    trainer = _trainer()
    state = create_river_game(("As", "Kd", "Qh", "Jc", "2s"), ("Ts", "3d"), ("9h", "9c"))
    store = StrategyStore(tmp_path / "strategies.sqlite3")
    written = store_river_root_strategy(store, trainer, state, Player.OOP, range_profile_id="sb-vs-bb", quality=analyze_river_profile(trainer))
    found = store.lookup_river(
        board=state.board,
        hero_position=Player.OOP,
        hero_cards=("3d", "Ts"),
        pot_bb=10,
        effective_stack_bb=95,
        range_profile_id="sb-vs-bb",
    )
    assert found is not None
    assert found.strategy_key == written.strategy_key
    assert found.quality is not None
    assert found.quality["nash_conv_bb"] >= 0
    assert sum(action.probability for action in found.actions) == 1.0
    generic = store.lookup(
        StrategyContext(
            game_type="river_heads_up", street="river", player_count=2, hero_position="oop", hero_cards=("3d", "Ts"),
            board=state.board, pot_units=1000, effective_stack_units=9500, action_history=(),
            range_profile_id="sb-vs-bb", solver_version="river-v1",
        )
    )
    assert generic is not None
    assert generic.actions == found.actions


def test_heads_up_flop_and_turn_infosets_export_to_generic_strategy_store(tmp_path):
    ranges = WeightedRange.from_cards((("Ts", "3d"),))
    opponent = WeightedRange.from_cards((("9h", "9c"),))
    policy = SizingPolicy(bet_sizes=(0.5,), raise_sizes=(), include_all_in=False, max_re_raises=0)
    store = StrategyStore(tmp_path / "strategies.sqlite3")

    flop = FlopMCCFRTrainer(flop_board=("As", "Kd", "Qh"), oop_range=ranges, ip_range=opponent, sizing_policy=policy, seed=1)
    flop.train(2)
    flop_state = create_flop_game(("As", "Kd", "Qh"), ("Ts", "3d"), ("9h", "9c"))
    store_heads_up_postflop_root_strategy(store, flop, flop_state, Player.OOP, game_type="flop_heads_up")
    flop_report = export_heads_up_postflop_infosets(store, flop, game_type="flop_heads_up")
    assert flop_report.stored_infosets > 0
    assert store.lookup(StrategyContext(
        game_type="flop_heads_up", street="flop", player_count=2, hero_position="oop", hero_cards=("3d", "Ts"),
        board=("As", "Kd", "Qh"), pot_units=1000, effective_stack_units=9500, action_history=(),
        range_profile_id="default", solver_version="heads-up-postflop-v1",
    )) is not None

    turn = TurnMCCFRTrainer(turn_board=("As", "Kd", "Qh", "Jc"), oop_range=ranges, ip_range=opponent, sizing_policy=policy, seed=1)
    turn.train(2)
    turn_state = create_turn_game(("As", "Kd", "Qh", "Jc"), ("Ts", "3d"), ("9h", "9c"))
    store_heads_up_postflop_root_strategy(store, turn, turn_state, Player.OOP, game_type="turn_heads_up")
    turn_report = export_heads_up_postflop_infosets(store, turn, game_type="turn_heads_up")
    assert turn_report.stored_infosets > 0


def test_batch_builder_accepts_flop_and_turn_heads_up_jobs(tmp_path):
    raw = json.loads((Path(__file__).resolve().parents[2] / "configs" / "heads_up_solution_grid.json").read_text(encoding="utf-8"))
    raw["max_canonical_boards_per_street"] = 1
    pack_dir = tmp_path / "heads_up_packs"
    jobs = build_heads_up_jobs(raw, pack_dir)
    pack_dir.mkdir()
    selected = [item for item in jobs if item[0]["game_type"] in {"flop_heads_up", "turn_heads_up"}][:2]
    for job, config in selected:
        (pack_dir / Path(job["config"]).name).write_text(json.dumps(config), encoding="utf-8")
    base = tmp_path
    store = StrategyStore(tmp_path / "strategies.sqlite3")
    parser = ArgumentParser()
    for job, _ in selected:
        _run_job(store, base, {
            "game_type": job["game_type"],
            "config": job["config"],
            "iterations": 1,
            "checkpoint": str(tmp_path / f"{job['game_type']}.pkl"),
            "checkpoint_every": 1,
            "range_profile_id": "test",
            "export_all_routes": True,
        }, 1, 1, parser)


def test_river_strategy_store_returns_none_for_untrained_context(tmp_path):
    store = StrategyStore(tmp_path / "strategies.sqlite3")
    assert store.lookup_river(
        board=("As", "Kd", "Qh", "Jc", "2s"),
        hero_position=Player.OOP,
        hero_cards=("Ts", "3d"),
        pot_bb=10,
        effective_stack_bb=95,
        range_profile_id="missing",
    ) is None


def test_river_strategy_key_changes_when_range_profile_changes():
    shared = dict(
        board=("As", "Kd", "Qh", "Jc", "2s"), hero_position="oop", hero_cards=("Ts", "3d"),
        pot_units=1000, effective_stack_units=9500, solver_version="river-v1",
    )
    assert river_strategy_key(**shared, range_profile_id="a") != river_strategy_key(**shared, range_profile_id="b")


def test_river_tree_export_reports_coverage_and_stores_non_root_routes(tmp_path):
    trainer = _trainer()
    store = StrategyStore(tmp_path / "strategies.sqlite3")
    report = export_river_tree(store, trainer, range_profile_id="tree-test")
    assert report.reachable_infosets == report.stored_infosets == len(trainer.infosets)
    assert report.unvisited_infosets == 0


def test_strategy_store_batch_reuses_one_connection_until_export_finishes(tmp_path):
    store = StrategyStore(tmp_path / "strategies.sqlite3")
    assert store._batch_connection is None
    with store.batch():
        connection = store._batch_connection
        assert connection is not None
        with store.batch():
            assert store._batch_connection is connection
    assert store._batch_connection is None


def test_strategy_store_buffered_upserts_uses_executemany_and_reports_progress(tmp_path):
    class Progress:
        total = 0

        def update(self, count):
            self.total += count

    store = StrategyStore(tmp_path / "strategies.sqlite3")
    progress = Progress()

    def record(name):
        context = StrategyContext("test", "river", 2, name, ("As", "Kd"), ("2c",), 100, 1000, (), "test", "v1")
        return StoredStrategy(strategy_key(context), context, 1, (StoredAction("check", None, "check", 1.0),), None)

    with store.batch(), store.buffered_upserts(batch_size=1, progress=progress):
        store.upsert(record("first"))
        store.upsert(record("second"))

    assert progress.total == 2
    assert store.lookup(record("first").context) is not None
    with store.buffered_upserts():
        with store.buffered_upserts():
            pass
    with pytest.raises(ValueError, match="batch_size"):
        with store.buffered_upserts(batch_size=0):
            pass
    with pytest.raises(RuntimeError):
        with store.batch(), store.buffered_upserts():
            store.upsert(record("discarded"))
            raise RuntimeError("中斷匯出")
    assert store.lookup(record("discarded").context) is None


def test_strategy_store_records_completed_pack_exports(tmp_path):
    store = StrategyStore(tmp_path / "strategies.sqlite3")
    assert not store.is_pack_export_complete("pack-1", trained_iterations=10, infoset_count=20)
    store.mark_pack_export_complete("pack-1", trained_iterations=10, infoset_count=20)
    assert store.is_pack_export_complete("pack-1", trained_iterations=10, infoset_count=20)
    assert not store.is_pack_export_complete("pack-1", trained_iterations=11, infoset_count=20)


def test_preflop_and_multiway_roots_share_the_generic_strategy_store(tmp_path):
    cards = (("As", "Kd"), ("Qh", "Jc"), ("Ts", "9d"), ("8h", "7c"), ("6s", "5d"), ("4h", "3c"), ("2s", "Ac"), ("Kh", "Qd"))
    ranges = {position: WeightedRange.from_cards((cards[index],)) for index, position in enumerate(Position)}
    store = StrategyStore(tmp_path / "strategies.sqlite3")
    preflop = MultiwayPreflopMCCFRTrainer(ranges=ranges, sizing_policy=PreflopSizingPolicy(include_all_in=False, max_raises=0), seed=3)
    preflop.train(1)
    root = create_8max_preflop(stack_bb=preflop.stack_bb)
    pre_record = store_preflop_root_strategy(store, preflop, root, Position.UTG, cards[0], range_profile_id="preflop-test")
    assert store.lookup(pre_record.context) is not None

    state = create_8max_preflop()
    for _ in range(6):
        state = apply_action(state, Action(ActionType.FOLD))
    state = apply_action(state, Action(ActionType.CALL))
    state = apply_action(state, Action(ActionType.CHECK))
    flop = advance_preflop_to_flop(state, ("Ah", "Kc", "Qc"))
    multiway = MultiwayPostflopMCCFRTrainer(
        initial_state=flop,
        ranges=ranges,
        sizing_policy=MultiwayPostflopSizingPolicy(bet_sizes=(0.5,), raise_sizes=(), include_all_in=False, max_re_raises=0),
        seed=4,
    )
    multiway.train(1)
    multi_record = store_multiway_postflop_root_strategy(store, multiway, flop, Position.SB, cards[6], range_profile_id="multiway-test")
    assert store.lookup(multi_record.context) is not None
    assert export_preflop_infosets(store, preflop, range_profile_id="preflop-all").stored_infosets == len(preflop.infosets)
    with store.batch():
        report = export_multiway_postflop_infosets(store, multiway, range_profile_id="multiway-all")
    assert report.stored_infosets == len(multiway.infosets)
    root_report = export_multiway_postflop_root_infosets(store, multiway, range_profile_id="multiway-root")
    assert 0 < root_report.stored_infosets < len(multiway.infosets)


def test_multiway_export_uses_remaining_players_not_all_eight_seats(tmp_path):
    cards = tuple((f"{rank}c", f"{rank}d") for rank in "23456789")
    ranges = {position: WeightedRange.from_cards((cards[index],)) for index, position in enumerate(Position)}
    state = create_8max_preflop()
    for _ in range(6):
        state = apply_action(state, Action(ActionType.FOLD))
    state = apply_action(state, Action(ActionType.CALL))
    state = apply_action(state, Action(ActionType.CHECK))
    trainer = MultiwayPostflopMCCFRTrainer(initial_state=advance_preflop_to_flop(state, ("As", "Kh", "Qh")), ranges=ranges, sizing_policy=MultiwayPostflopSizingPolicy(bet_sizes=(0.5,), raise_sizes=(), include_all_in=False, max_re_raises=0), seed=1)
    trainer.train(1)
    store = StrategyStore(tmp_path / "strategies.sqlite3")
    export_multiway_postflop_infosets(store, trainer, range_profile_id="active-count")
    key = next(iter(trainer.infosets))
    position, hole_cards, street, board, pot, _bet, _raise, _seats, _pending, _raise_allowed, history = key
    found = store.lookup(StrategyContext("multiway_postflop", street, 2, position, tuple(sorted(hole_cards)), tuple(board), pot, max(player.stack for player in trainer.initial_state.players), tuple(history), "active-count", "multiway-postflop-v1"))
    assert found is not None
