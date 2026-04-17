"""Dragon Gate card drawing simulator.

Set the number of decks (minimum one), draw two-card pairs without replacement,
and reset or reshuffle the sample space manually when needed.
"""

from __future__ import annotations

import random

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence, Tuple

import ipywidgets as widgets
from IPython.display import clear_output, display


RANKS: Sequence[str] = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
SUITS: Sequence[str] = ["S", "H", "D", "C"]
SECTION_LABELS: Sequence[str] = ["1~3", "4~6", "7~9", "10~13"]
FOUR_BUCKETS: Sequence[Tuple[int, ...]] = (
    (1, 2, 3),
    (4, 5, 6),
    (7, 8, 9),
    (10, 11, 12, 13),
)
DEFAULT_PAYOUTS = {
    "inside": 1.0,
    "outside": -1.0,
    "edge": -1.0,
    "same_rank_pair_hit": 3.0,
    "same_rank_pair_miss": 0.0,
}


@dataclass(frozen=True)
class Card:
    """Represents a single card (suit + rank)."""

    rank: str
    suit: str

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"


@dataclass(frozen=True)
class RoundOutcome:
    """Represents the round result after the third card is revealed."""

    key: str
    label: str
    payout: float


@dataclass(frozen=True)
class GateEvaluation:
    """Represents four-bucket evaluation scores for a gate."""

    c_in: dict[str, float]
    c_out: dict[str, float]
    S_in: float
    S_out: float
    decision: str


@dataclass(frozen=True)
class ExactExpectation:
    """Represents exact expected value for betting inside or outside."""

    inside_ev: float
    outside_ev: float
    probabilities: dict[str, float]


class DragonGateDeck:
    """Manage the Dragon Gate deck pool.

    Uses multiple standard 52-card decks. Each draw returns a pair (two cards)
    and the cards are not returned unless reset or reshuffle is triggered.
    """

    def __init__(self, decks: int = 1, seed: int | None = None) -> None:
        if decks < 1:
            raise ValueError("至少要有 1 副牌。")
        self._rng = random.Random(seed)
        self._deck_size = decks
        self._base_cards = tuple(Card(rank, suit) for suit in SUITS for rank in RANKS)
        self._discard: List[Card] = []
        self.reset()

    def reset(self) -> None:
        """Reset and shuffle the deck to start a new round."""

        cards = list(self._base_cards) * self._deck_size
        self._rng.shuffle(cards)
        self._cards = cards
        self._discard.clear()

    def reshuffle(self) -> None:
        """Return discarded cards to the deck and shuffle them."""

        self._cards.extend(self._discard)
        self._discard.clear()
        self._rng.shuffle(self._cards)

    def draw_pair(self, auto_reset: bool = True) -> Tuple[Card, Card]:
        """Draw two cards. The deck resets automatically when empty by default.

        Args:
            auto_reset: automatically reset when insufficient cards remain.

        Raises:
            RuntimeError: when cards are insufficient and auto reset is disabled.
        """

        if len(self._cards) < 2:
            if auto_reset:
                self.reset()
            else:
                raise RuntimeError("牌數不足且未允許自動重新洗牌。")

        drawn = (self._cards.pop(), self._cards.pop())
        self._discard.extend(drawn)
        return drawn

    def draw_one(self, auto_reset: bool = True) -> Card:
        """Draw a single card from the deck."""

        if len(self._cards) < 1:
            if auto_reset:
                self.reset()
            else:
                raise RuntimeError("牌數不足且未允許自動重新洗牌。")

        drawn = self._cards.pop()
        self._discard.append(drawn)
        return drawn

    def draw_multiple_pairs(
        self, pair_count: int, auto_reset: bool = True
    ) -> Iterator[Tuple[Card, Card]]:
        """Draw multiple sequential pairs."""

        for _ in range(pair_count):
            yield self.draw_pair(auto_reset=auto_reset)

    @property
    def remaining(self) -> int:
        """Number of cards currently left in the deck."""

        return len(self._cards)

    @property
    def discarded(self) -> int:
        """Number of cards that have been discarded."""

        return len(self._discard)

    def sample_space(self) -> Sequence[Card]:
        """Return a copy of all cards that are still in the deck."""

        return tuple(self._cards)

    def __iter__(self) -> Iterator[Card]:
        """Allow iteration over the remaining cards."""

        return iter(self._cards)


def build_deck(decks: int = 1, seed: int | None = None) -> DragonGateDeck:
    """Convenience factory for creating a simulator in notebooks or scripts."""

    return DragonGateDeck(decks=decks, seed=seed)


def draw_until_empty(deck: DragonGateDeck) -> List[Tuple[Card, Card]]:
    """Example helper that draws every remaining pair without reshuffling."""

    pairs: List[Tuple[Card, Card]] = []
    while deck.remaining >= 2:
        pairs.append(deck.draw_pair(auto_reset=False))
    return pairs


def _rank_to_section(rank: str) -> int:
    rank_value = RANKS.index(rank) + 1
    return min((rank_value - 1) // 3, len(SECTION_LABELS) - 1)


def _count_sections(cards: Sequence[Card]) -> List[int]:
    counts = [0] * len(SECTION_LABELS)
    for card in cards:
        counts[_rank_to_section(card.rank)] += 1
    return counts


def _count_remaining_ranks(cards: Sequence[Card]) -> dict[int, int]:
    counts = {rank_value: 0 for rank_value in range(1, 14)}
    for card in cards:
        counts[_rank_value(card.rank)] += 1
    return counts


def _rank_value(rank: str) -> int:
    return RANKS.index(rank) + 1


def _format_payout(payout: float) -> str:
    if payout > 0:
        return f"+{payout:g}"
    return f"{payout:g}"


def _display_card(card: Card) -> str:
    return card.rank


def _normalize_gate(a: int, b: int) -> Tuple[int, int]:
    if not 1 <= a <= 13 or not 1 <= b <= 13:
        raise ValueError("門牌點數必須介於 1 到 13。")
    return (a, b) if a < b else (b, a)


def _classify_rank(rank_value: int, a: int, b: int) -> str:
    if rank_value == a or rank_value == b:
        return "pillar"
    if a < rank_value < b:
        return "inside"
    return "outside"


def _validate_bucket_counts(bucket_counts: Sequence[float]) -> None:
    if len(bucket_counts) != len(FOUR_BUCKETS):
        raise ValueError("四分法需要剛好四個 bucket 權重。")


def evaluate_gate(
    a: int, b: int, R1: float, R2: float, R3: float, R4: float
) -> GateEvaluation:
    """Evaluate a gate using the four-bucket approximation.

    R1..R4 are the remaining weights/counts for buckets [1,2,3], [4,5,6],
    [7,8,9], [10,11,12,13].
    """

    low, high = _normalize_gate(a, b)
    bucket_weights = (R1, R2, R3, R4)
    _validate_bucket_counts(bucket_weights)

    c_in: dict[str, float] = {}
    c_out: dict[str, float] = {}
    S_in = 0.0
    S_out = 0.0

    for label, bucket, weight in zip(SECTION_LABELS, FOUR_BUCKETS, bucket_weights):
        inside_count = 0
        pillar_count = 0
        outside_count = 0

        for rank_value in bucket:
            classification = _classify_rank(rank_value, low, high)
            if classification == "inside":
                inside_count += 1
            elif classification == "pillar":
                pillar_count += 1
            else:
                outside_count += 1

        bucket_size = len(bucket)
        inside_coeff = (
            inside_count - outside_count - 2 * pillar_count
        ) / bucket_size
        outside_coeff = (
            outside_count - inside_count - 2 * pillar_count
        ) / bucket_size

        c_in[label] = inside_coeff
        c_out[label] = outside_coeff
        S_in += inside_coeff * weight
        S_out += outside_coeff * weight

    if S_in > 0 and S_in >= S_out:
        decision = "inside"
    elif S_out > 0 and S_out > S_in:
        decision = "outside"
    else:
        decision = "skip"

    return GateEvaluation(
        c_in=c_in,
        c_out=c_out,
        S_in=S_in,
        S_out=S_out,
        decision=decision,
    )


def exact_expected_value(
    a: int,
    b: int,
    remaining_rank_counts: dict[int, int],
    payouts: dict[str, float] | None = None,
) -> ExactExpectation:
    """Calculate exact EV from the remaining counts of ranks 1..13."""

    low, high = _normalize_gate(a, b)
    payout_table = DEFAULT_PAYOUTS if payouts is None else payouts

    total_cards = sum(remaining_rank_counts.values())
    if total_cards <= 0:
        raise ValueError("remaining_rank_counts 必須包含至少一張牌。")

    inside_cards = 0
    pillar_cards = 0
    outside_cards = 0

    for rank_value, count in remaining_rank_counts.items():
        if count < 0:
            raise ValueError("remaining_rank_counts 不能有負數。")
        classification = _classify_rank(rank_value, low, high)
        if classification == "inside":
            inside_cards += count
        elif classification == "pillar":
            pillar_cards += count
        else:
            outside_cards += count

    inside_prob = inside_cards / total_cards
    pillar_prob = pillar_cards / total_cards
    outside_prob = outside_cards / total_cards

    inside_ev = (
        inside_prob * payout_table["inside"]
        + pillar_prob * payout_table["edge"]
        + outside_prob * payout_table["outside"]
    )
    outside_ev = (
        outside_prob * abs(payout_table["outside"])
        - inside_prob * payout_table["inside"]
        + pillar_prob * payout_table["edge"]
    )

    return ExactExpectation(
        inside_ev=inside_ev,
        outside_ev=outside_ev,
        probabilities={
            "inside": inside_prob,
            "pillar": pillar_prob,
            "outside": outside_prob,
        },
    )


def _evaluate_round(
    pair: Tuple[Card, Card], third_card: Card, payouts: dict[str, float]
) -> RoundOutcome:
    first_value = _rank_value(pair[0].rank)
    second_value = _rank_value(pair[1].rank)
    third_value = _rank_value(third_card.rank)

    if first_value == second_value:
        if third_value == first_value:
            return RoundOutcome(
                key="same_rank_pair_hit",
                label="龍門同點，第三張同點，判定撞柱。",
                payout=payouts["same_rank_pair_hit"],
            )
        return RoundOutcome(
            key="same_rank_pair_miss",
            label="龍門兩張同點，第三張不同點，這局和局處理。",
            payout=payouts["same_rank_pair_miss"],
        )

    low_value = min(first_value, second_value)
    high_value = max(first_value, second_value)
    if third_value == low_value or third_value == high_value:
        return RoundOutcome(
            key="edge",
            label="第三張撞柱。",
            payout=payouts["edge"],
        )
    if low_value < third_value < high_value:
        return RoundOutcome(
            key="inside",
            label="第三張落在龍門內，中龍門。",
            payout=payouts["inside"],
        )
    return RoundOutcome(
        key="outside",
        label="第三張落在龍門外，未中龍門。",
        payout=payouts["outside"],
    )


def launch_interactive_simulation(deck_count: int = 2, seed: int | None = 20260414) -> None:
    """Display the interactive simulation controls inside a notebook."""

    deck = build_deck(decks=deck_count, seed=seed)
    draw_history: List[Tuple[Card, Card]] = []
    section_counts = _count_sections(deck.sample_space())
    current_pair: Tuple[Card, Card] | None = None
    current_third_card: Card | None = None
    current_result = "尚未開局。"
    current_outcome: RoundOutcome | None = None
    payouts = DEFAULT_PAYOUTS.copy()

    status_out = widgets.Output()
    draw_out = widgets.Output()
    section_out = widgets.Output()
    history_out = widgets.Output()
    four_method_out = widgets.Output()
    exact_ev_out = widgets.Output()
    section_panel = widgets.VBox([section_out], layout=widgets.Layout(display="none"))
    history_panel = widgets.VBox([history_out], layout=widgets.Layout(display="none"))
    four_method_panel = widgets.VBox([four_method_out], layout=widgets.Layout(display="none"))
    exact_ev_panel = widgets.VBox([exact_ev_out], layout=widgets.Layout(display="none"))

    def update_status() -> None:
        with status_out:
            clear_output()
            print(f"底牌剩 {deck.remaining} 張、棄牌 {deck.discarded} 張。")
            if current_pair is None:
                print("目前龍門：尚未抽兩張底牌。")
            else:
                print(f"目前龍門：{_display_card(current_pair[0])} & {_display_card(current_pair[1])}")
                if current_third_card is None:
                    print("第三張底牌：尚未翻開。")
                else:
                    print(f"第三張底牌：{_display_card(current_third_card)}")
                print(f"判定：{current_result}")
            if deck.remaining < 2:
                print("< 2 張，請先重洗或重新起始牌局。")

    def update_sections() -> None:
        with section_out:
            clear_output()
            section_display = " / ".join(
                f"{SECTION_LABELS[idx]}: {section_counts[idx]}"
                for idx in range(len(SECTION_LABELS))
            )
            print(f"各段剩餘：{section_display}")

    def update_history() -> None:
        with history_out:
            clear_output()
            print("抽牌紀錄：")
            if not draw_history:
                print("  - 尚未抽牌")
                return
            for idx, pair in enumerate(draw_history, 1):
                print(f"  {idx:2d}: {_display_card(pair[0])} & {_display_card(pair[1])}")

    def update_four_method() -> None:
        with four_method_out:
            clear_output()
            if current_pair is None:
                print("四分法：尚未抽龍門。")
                return

            evaluation = evaluate_gate(
                _rank_value(current_pair[0].rank),
                _rank_value(current_pair[1].rank),
                *section_counts,
            )
            print("四分法：")
            print(
                "  c_in = "
                + ", ".join(f"{label}:{evaluation.c_in[label]:.3f}" for label in SECTION_LABELS)
            )
            print(
                "  c_out = "
                + ", ".join(f"{label}:{evaluation.c_out[label]:.3f}" for label in SECTION_LABELS)
            )
            print(f"  S_in = {evaluation.S_in:.3f}")
            print(f"  S_out = {evaluation.S_out:.3f}")
            print(f"  decision = {evaluation.decision}")

    def update_exact_ev() -> None:
        with exact_ev_out:
            clear_output()
            if current_pair is None:
                print("精確期望值：尚未抽龍門。")
                return

            remaining_rank_counts = _count_remaining_ranks(deck.sample_space())
            expectation = exact_expected_value(
                _rank_value(current_pair[0].rank),
                _rank_value(current_pair[1].rank),
                remaining_rank_counts,
                payouts,
            )
            print("精確期望值：")
            print(f"  inside_ev = {expectation.inside_ev:.6f}")
            print(f"  outside_ev = {expectation.outside_ev:.6f}")
            print(
                "  probabilities = "
                f"inside:{expectation.probabilities['inside']:.6f}, "
                f"pillar:{expectation.probabilities['pillar']:.6f}, "
                f"outside:{expectation.probabilities['outside']:.6f}"
            )

    def _decrement_section(card: Card) -> None:
        section_counts[_rank_to_section(card.rank)] -= 1

    def draw_pair_once(_: widgets.Button) -> None:
        nonlocal current_pair, current_third_card, current_result, current_outcome
        with draw_out:
            clear_output()
            if current_pair is not None and current_third_card is None:
                print("請先翻開第三張底牌，再抽下一組龍門。")
                return
            try:
                pair = deck.draw_pair(auto_reset=False)
            except RuntimeError as exc:
                print(exc)
                return
            draw_history.append(pair)
            current_pair = pair
            current_third_card = None
            current_result = "等待翻開第三張底牌。"
            current_outcome = None
            print(
                f"第 {len(draw_history)} 組龍門："
                f"{_display_card(pair[0])} & {_display_card(pair[1])}"
            )
        _decrement_section(pair[0])
        _decrement_section(pair[1])
        reveal_button.disabled = False
        update_status()
        update_sections()
        update_history()
        update_four_method()
        update_exact_ev()

    def reveal_third_card(_: widgets.Button) -> None:
        nonlocal current_third_card, current_result, current_outcome
        with draw_out:
            clear_output()
            if current_pair is None:
                print("請先抽兩張底牌形成龍門。")
                return
            if current_third_card is not None:
                print("這組龍門已經翻過第三張底牌，請抽下一組。")
                return
            try:
                current_third_card = deck.draw_one(auto_reset=False)
            except RuntimeError as exc:
                print(exc)
                return
            _decrement_section(current_third_card)
            current_outcome = _evaluate_round(current_pair, current_third_card, payouts)
            current_result = current_outcome.label
            print(f"第三張底牌：{_display_card(current_third_card)}")
            print(current_result)
        reveal_button.disabled = True
        update_status()
        update_sections()
        update_four_method()
        update_exact_ev()

    def reshuffle_history(_: widgets.Button) -> None:
        nonlocal section_counts, current_pair, current_third_card, current_result, current_outcome
        deck.reshuffle()
        section_counts = _count_sections(deck.sample_space())
        current_pair = None
        current_third_card = None
        current_result = "已洗回棄牌，等待重新抽龍門。"
        current_outcome = None
        with draw_out:
            clear_output()
            print("已回收棄牌並重洗。")
        reveal_button.disabled = True
        update_status()
        update_sections()
        update_four_method()
        update_exact_ev()

    def reset_history(_: widgets.Button) -> None:
        nonlocal section_counts, current_pair, current_third_card, current_result, current_outcome
        deck.reset()
        draw_history.clear()
        section_counts = _count_sections(deck.sample_space())
        current_pair = None
        current_third_card = None
        current_result = "尚未開局。"
        current_outcome = None
        with draw_out:
            clear_output()
            print("✅ 從頭再來：重新洗牌並清除紀錄。")
        reveal_button.disabled = True
        update_status()
        update_sections()
        update_history()
        update_four_method()
        update_exact_ev()

    def toggle_history(change) -> None:
        visible = change["new"]
        history_panel.layout.display = "flex" if visible else "none"
        history_button.description = "隱藏抽牌紀錄" if visible else "顯示抽牌紀錄"
        if visible:
            update_history()

    def toggle_sections(change) -> None:
        visible = change["new"]
        section_panel.layout.display = "flex" if visible else "none"
        section_button.description = "隱藏各段剩餘" if visible else "顯示各段剩餘"
        if visible:
            update_sections()

    def toggle_four_method(change) -> None:
        visible = change["new"]
        four_method_panel.layout.display = "flex" if visible else "none"
        four_method_button.description = "隱藏四分法" if visible else "顯示四分法"
        if visible:
            update_four_method()

    def toggle_exact_ev(change) -> None:
        visible = change["new"]
        exact_ev_panel.layout.display = "flex" if visible else "none"
        exact_ev_button.description = "隱藏精確期望值" if visible else "顯示精確期望值"
        if visible:
            update_exact_ev()

    draw_button = widgets.Button(description="抽兩張")
    reveal_button = widgets.Button(description="翻第三張底牌", disabled=True)
    reshuffle_button = widgets.Button(description="洗回棄牌並洗牌")
    reset_button = widgets.Button(description="從頭再來")
    section_button = widgets.ToggleButton(
        description="顯示各段剩餘", value=False, tooltip="按下後才會顯示各段剩餘"
    )
    four_method_button = widgets.ToggleButton(
        description="顯示四分法", value=False, tooltip="按下後才會顯示四分法"
    )
    exact_ev_button = widgets.ToggleButton(
        description="顯示精確期望值", value=False, tooltip="按下後才會顯示精確期望值"
    )
    history_button = widgets.ToggleButton(
        description="顯示抽牌紀錄", value=False, tooltip="按下後才會顯示抽牌紀錄"
    )
    draw_button.on_click(draw_pair_once)
    reveal_button.on_click(reveal_third_card)
    reshuffle_button.on_click(reshuffle_history)
    reset_button.on_click(reset_history)
    section_button.observe(toggle_sections, names="value")
    four_method_button.observe(toggle_four_method, names="value")
    exact_ev_button.observe(toggle_exact_ev, names="value")
    history_button.observe(toggle_history, names="value")

    controls = widgets.HBox(
        [
            draw_button,
            reveal_button,
            reshuffle_button,
            reset_button,
            section_button,
            four_method_button,
            exact_ev_button,
            history_button,
        ],
        layout=widgets.Layout(flex_flow="row wrap", gap="12px"),
    )
    display(
        widgets.VBox(
            [
                controls,
                status_out,
                draw_out,
                section_panel,
                four_method_panel,
                exact_ev_panel,
                history_panel,
            ]
        )
    )
    update_status()
    update_sections()
    update_history()
    update_four_method()
    update_exact_ev()
