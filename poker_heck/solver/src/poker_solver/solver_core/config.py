"""River solver JSON 設定檔讀取。"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from poker_solver.engine.river_game import SizingPolicy
from poker_solver.solver_core.range_spec import expand_range_spec
from poker_solver.solver_core.river_mccfr import Combo, RiverMCCFRTrainer, TurnRiverMCCFRTrainer, WeightedRange
from poker_solver.solver_core.turn_mccfr import FlopMCCFRTrainer, TurnMCCFRTrainer


@dataclass(frozen=True)
class RiverSolveConfig:
    board: tuple[str, ...]
    oop_range: WeightedRange
    ip_range: WeightedRange
    sizing_policy: SizingPolicy
    initial_pot_bb: int | float | str = 10
    effective_stack_bb: int | float | str = 95
    seed: int = 0
    mode: str = "river"

    def create_trainer(self) -> RiverMCCFRTrainer | TurnMCCFRTrainer | FlopMCCFRTrainer:
        if len(self.board) == 3:
            return FlopMCCFRTrainer(
                flop_board=self.board,  # type: ignore[arg-type]
                oop_range=self.oop_range,
                ip_range=self.ip_range,
                sizing_policy=self.sizing_policy,
                initial_pot_bb=self.initial_pot_bb,
                effective_stack_bb=self.effective_stack_bb,
                seed=self.seed,
            )
        if len(self.board) == 4:
            if self.mode == "turn":
                return TurnMCCFRTrainer(
                    turn_board=self.board,  # type: ignore[arg-type]
                    oop_range=self.oop_range,
                    ip_range=self.ip_range,
                    sizing_policy=self.sizing_policy,
                    initial_pot_bb=self.initial_pot_bb,
                    effective_stack_bb=self.effective_stack_bb,
                    seed=self.seed,
                )
            return TurnRiverMCCFRTrainer(
                turn_board=self.board,  # type: ignore[arg-type]
                oop_range=self.oop_range,
                ip_range=self.ip_range,
                sizing_policy=self.sizing_policy,
                initial_pot_bb=self.initial_pot_bb,
                effective_stack_bb=self.effective_stack_bb,
                seed=self.seed,
            )
        return RiverMCCFRTrainer(
            board=self.board,  # type: ignore[arg-type]
            oop_range=self.oop_range,
            ip_range=self.ip_range,
            sizing_policy=self.sizing_policy,
            initial_pot_bb=self.initial_pot_bb,
            effective_stack_bb=self.effective_stack_bb,
            seed=self.seed,
        )


def load_config(path: str | Path) -> RiverSolveConfig:
    """載入 River MCCFR 設定檔。"""
    with Path(path).open(encoding="utf-8") as file:
        raw = json.load(file)
    try:
        board = tuple(raw["board"])
        if len(board) not in {3, 4, 5}:
            raise ValueError("board must contain three (flop), four (turn), or five (river) cards")
        sizing = raw.get("sizing_policy", {})
        policy = SizingPolicy(
            bet_sizes=tuple(sizing.get("bet_sizes", (0.33, 0.50, 0.75, 1.0, 1.5, 2.0))),
            raise_sizes=tuple(sizing.get("raise_sizes", (0.33, 0.50, 0.75, 1.0, 1.5, 2.0))),
            include_all_in=bool(sizing.get("include_all_in", True)),
            max_re_raises=sizing.get("max_re_raises", 2),
        )
        return RiverSolveConfig(
            board=board,  # type: ignore[arg-type]
            oop_range=_parse_range(raw, "oop", board),
            ip_range=_parse_range(raw, "ip", board),
            sizing_policy=policy,
            initial_pot_bb=raw.get("initial_pot_bb", 10),
            effective_stack_bb=raw.get("effective_stack_bb", 95),
            seed=int(raw.get("seed", 0)),
            mode=str(raw.get("mode", "river")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid river solve configuration: {error}") from error


def _parse_range(raw: dict[str, Any], player: str, board: tuple[str, ...]) -> WeightedRange:
    if f"{player}_range_spec" in raw:
        return expand_range_spec(raw[f"{player}_range_spec"], excluded_cards=board)
    raw_range = raw[f"{player}_range"]
    if not isinstance(raw_range, list):
        raise ValueError("range must be a list")
    combos = []
    for entry in raw_range:
        cards = tuple(entry["cards"])
        if len(cards) != 2:
            raise ValueError("each range combo must contain two cards")
        combos.append(Combo(cards, float(entry.get("weight", 1.0))))  # type: ignore[arg-type]
    return WeightedRange(tuple(combos))
