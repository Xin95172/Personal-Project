"""Heads-up river 平均策略的精確樹狀評估工具。

River 沒有未發公共牌，因此可枚舉已訓練 range 的所有相容 combo，精確算出
平均策略 EV 與固定對手策略下的 best response。這個數字只適用於目前的
動作抽象與 ranges；它不是跨範圍、連 flop/turn 的完整 exploitability。
"""

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from poker_solver.engine.money import bb_to_units
from poker_solver.engine.river_game import Action, Player, RiverGameState, abstract_actions, apply_action, create_river_game, infoset_key, is_terminal, terminal_utility
from poker_solver.solver_core.river_mccfr import Combo, RiverMCCFRTrainer


@dataclass(frozen=True)
class RiverQualityReport:
    profile_ev_oop_bb: float
    oop_best_response_bb: float
    ip_best_response_bb: float
    nash_conv_bb: float
    nash_conv_initial_pot_fraction: float
    nash_conv_effective_stack_fraction: float
    compatible_combo_pairs: int


def write_river_quality_report(report: RiverQualityReport, path: str | Path) -> Path:
    """將可供 CI 或後續比較的 river 品質指標寫成 JSON。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "scope": "heads-up river; trained ranges and action abstraction only",
                "profile_ev_oop_bb": report.profile_ev_oop_bb,
                "oop_best_response_bb": report.oop_best_response_bb,
                "ip_best_response_bb": report.ip_best_response_bb,
                "nash_conv_bb": report.nash_conv_bb,
                "nash_conv_initial_pot_fraction": report.nash_conv_initial_pot_fraction,
                "nash_conv_effective_stack_fraction": report.nash_conv_effective_stack_fraction,
                "nash_conv_initial_pot_percent": 100 * report.nash_conv_initial_pot_fraction,
                "nash_conv_effective_stack_percent": 100 * report.nash_conv_effective_stack_fraction,
                "compatible_combo_pairs": report.compatible_combo_pairs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return target


WeightedState = tuple[RiverGameState, float]


def analyze_river_profile(trainer: RiverMCCFRTrainer) -> RiverQualityReport:
    """在 trainer 的 river root game 上計算 profile EV 與 NashConv。"""
    pairs = tuple(_compatible_pairs(trainer))
    if not pairs:
        raise ValueError("the supplied ranges cannot produce a non-overlapping deal")
    total_weight = sum(weight for _, _, weight in pairs)
    normalized = tuple((oop, ip, weight / total_weight) for oop, ip, weight in pairs)
    profile_ev = _profile_value(trainer, _states_for_pairs(trainer, normalized), Player.OOP)

    oop_br = sum(
        _best_response_value(trainer, _states_for_pairs(trainer, group), Player.OOP)
        for group in _groups_by_private_combo(normalized, Player.OOP)
    )
    ip_br_for_ip = sum(
        _best_response_value(trainer, _states_for_pairs(trainer, group), Player.IP)
        for group in _groups_by_private_combo(normalized, Player.IP)
    )
    # 對零和 utility 而言，IP 的收益為 OOP utility 的相反數。
    ip_best_response_oop_utility = -ip_br_for_ip
    nash_conv_bb = (oop_br - ip_best_response_oop_utility) / 100
    initial_pot_bb = bb_to_units(trainer.initial_pot_bb) / 100
    effective_stack_bb = bb_to_units(trainer.effective_stack_bb) / 100
    return RiverQualityReport(
        profile_ev_oop_bb=profile_ev / 100,
        oop_best_response_bb=oop_br / 100,
        ip_best_response_bb=ip_br_for_ip / 100,
        nash_conv_bb=nash_conv_bb,
        nash_conv_initial_pot_fraction=nash_conv_bb / initial_pot_bb,
        nash_conv_effective_stack_fraction=nash_conv_bb / effective_stack_bb,
        compatible_combo_pairs=len(pairs),
    )


def _compatible_pairs(trainer: RiverMCCFRTrainer) -> Iterable[tuple[Combo, Combo, float]]:
    board = set(trainer.board)
    for oop in trainer.oop_range.combos:
        for ip in trainer.ip_range.combos:
            if len(board | set(oop.cards) | set(ip.cards)) == 9:
                yield oop, ip, oop.weight * ip.weight


def _states_for_pairs(trainer: RiverMCCFRTrainer, pairs: Iterable[tuple[Combo, Combo, float]]) -> tuple[WeightedState, ...]:
    return tuple(
        (
            create_river_game(trainer.board, oop.cards, ip.cards, initial_pot_bb=trainer.initial_pot_bb, effective_stack_bb=trainer.effective_stack_bb),
            weight,
        )
        for oop, ip, weight in pairs
    )


def _groups_by_private_combo(
    pairs: tuple[tuple[Combo, Combo, float], ...], player: Player
) -> Iterable[tuple[tuple[Combo, Combo, float], ...]]:
    grouped: dict[Combo, list[tuple[Combo, Combo, float]]] = defaultdict(list)
    for oop, ip, weight in pairs:
        grouped[oop if player is Player.OOP else ip].append((oop, ip, weight))
    return tuple(tuple(group) for group in grouped.values())


def _profile_value(trainer: RiverMCCFRTrainer, states: tuple[WeightedState, ...], perspective: Player) -> float:
    if not states:
        return 0.0
    if is_terminal(states[0][0]):
        return sum(weight * terminal_utility(state)[0 if perspective is Player.OOP else 1] for state, weight in states)
    actor = states[0][0].current_player
    assert actor is not None
    branches: dict[Action, list[WeightedState]] = defaultdict(list)
    for state, weight in states:
        for action, probability in _strategy(trainer, state, actor).items():
            branches[action].append((apply_action(state, action), weight * probability))
    return sum(_profile_value(trainer, tuple(branch), perspective) for branch in branches.values())


def _best_response_value(trainer: RiverMCCFRTrainer, states: tuple[WeightedState, ...], responder: Player) -> float:
    if not states:
        return 0.0
    if is_terminal(states[0][0]):
        index = 0 if responder is Player.OOP else 1
        return sum(weight * terminal_utility(state)[index] for state, weight in states)
    actor = states[0][0].current_player
    assert actor is not None
    actions = abstract_actions(states[0][0], trainer.sizing_policy)
    if actor is responder:
        return max(
            _best_response_value(trainer, tuple((apply_action(state, action), weight) for state, weight in states), responder)
            for action in actions
        )
    branches: dict[Action, list[WeightedState]] = defaultdict(list)
    for state, weight in states:
        for action, probability in _strategy(trainer, state, actor).items():
            branches[action].append((apply_action(state, action), weight * probability))
    return sum(_best_response_value(trainer, tuple(branch), responder) for branch in branches.values())


def _strategy(trainer: RiverMCCFRTrainer, state: RiverGameState, player: Player) -> dict[Action, float]:
    """未被 sampling 走訪的資訊集採 uniform，避免評估時改寫訓練資料。"""
    node = trainer.infosets.get(infoset_key(state, player))
    if node is not None:
        return node.average_strategy()
    actions = abstract_actions(state, trainer.sizing_policy)
    return {action: 1.0 / len(actions) for action in actions}
