"""讀取多人 postflop 訓練設定 JSON。"""

from __future__ import annotations

import json
from pathlib import Path

from poker_solver.engine.money import bb_to_units
from poker_solver.solver_core.range_spec import expand_range_spec
from poker_solver.engine.multiway_postflop_policy import MultiwayPostflopSizingPolicy
from poker_solver.engine.river_game import Action, ActionType
from poker_solver.engine.table import advance_preflop_to_flop, advance_multiway_postflop_street, apply_action, apply_multiway_postflop_action, create_8max_preflop, is_terminal
from poker_solver.solver_core.multiway_postflop_mccfr import MultiwayPostflopMCCFRTrainer
from poker_solver.solver_core.river_mccfr import Combo, WeightedRange
from poker_solver.engine.table import Position


def load_multiway_postflop_trainer(path: str | Path) -> MultiwayPostflopMCCFRTrainer:
    with Path(path).open(encoding="utf-8") as file:
        raw = json.load(file)
    try:
        preflop = create_8max_preflop(stack_bb=raw.get("stack_bb", 100))
        for entry in raw["preflop_actions"]:
            kind = ActionType(entry["kind"])
            amount = bb_to_units(entry["amount_bb"]) if "amount_bb" in entry else None
            preflop = apply_action(preflop, Action(kind, amount))
        if not is_terminal(preflop) or preflop.hand_ended:
            raise ValueError("preflop_actions must finish a non-fold preflop round")
        board = tuple(raw["board"])
        if len(board) not in {3, 4, 5}:
            raise ValueError("board must contain 3, 4, or 5 cards")
        state = advance_preflop_to_flop(preflop, board[:3])
        for next_card, street_actions in zip(board[3:], raw["completed_street_actions"]):
            for entry in street_actions:
                action = Action(ActionType(entry["kind"]), bb_to_units(entry["amount_bb"]) if "amount_bb" in entry else None)
                state = apply_multiway_postflop_action(state, action)
            if not state.betting_complete:
                raise ValueError("each completed_street_actions entry must finish its street")
            state = advance_multiway_postflop_street(state, next_card)
        if len(raw["completed_street_actions"]) != len(board) - 3:
            raise ValueError("completed_street_actions count must match board street")
        ranges = _load_ranges(raw, board)
        policy = raw.get("sizing_policy", {})
        return MultiwayPostflopMCCFRTrainer(
            initial_state=state,
            ranges=ranges,
            sizing_policy=MultiwayPostflopSizingPolicy(
                bet_sizes=tuple(policy.get("bet_sizes", (0.33, 0.50, 0.75, 1.0, 1.5, 2.0))),
                raise_sizes=tuple(policy.get("raise_sizes", (0.33, 0.50, 0.75, 1.0, 1.5, 2.0))),
                include_all_in=bool(policy.get("include_all_in", True)),
                max_re_raises=policy.get("max_re_raises", 2),
            ),
            seed=int(raw.get("seed", 0)),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid multiway postflop solve configuration: {error}") from error


def _load_ranges(raw: dict, board: tuple[str, ...]):
    if raw.get("range_spec", {}).get("kind") == "all_combos":
        full = expand_range_spec(raw["range_spec"], excluded_cards=board)
        return {position: full for position in Position}
    return {
            position: WeightedRange(
                tuple(Combo(tuple(entry["cards"]), float(entry.get("weight", 1.0))) for entry in raw["ranges"][position.value])  # type: ignore[arg-type]
            )
            for position in Position
        }
