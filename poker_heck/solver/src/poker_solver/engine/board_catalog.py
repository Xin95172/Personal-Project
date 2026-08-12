"""公共牌的花色同構（suit-isomorphic）catalog。"""

from itertools import combinations, permutations
from typing import Iterator


RANKS = "23456789TJQKA"
SUITS = "cdhs"
FULL_DECK = tuple(f"{rank}{suit}" for rank in RANKS for suit in SUITS)
_SUIT_PERMUTATIONS = tuple(permutations(SUITS))


def canonicalize_board(board: tuple[str, ...]) -> tuple[str, ...]:
    """將一組公共牌映射為唯一的花色同構代表。

    牌面順序不影響 solver 的起始資訊集；rank 保持不變，只枚舉 24 種
    花色置換並選字典序最小者作為 key。
    """
    if len(set(board)) != len(board) or any(len(card) != 2 or card[0] not in RANKS or card[1] not in SUITS for card in board):
        raise ValueError("board must contain distinct valid cards")
    return min(tuple(sorted(f"{card[0]}{mapping[SUITS.index(card[1])]}" for card in board)) for mapping in _SUIT_PERMUTATIONS)


def iter_canonical_boards(card_count: int, *, limit: int | None = None) -> Iterator[tuple[str, ...]]:
    """列舉所有 card_count 張公共牌的花色同構代表。

    ``limit`` 僅供分批／測試使用；省略時才是真正完整列舉。river 的完整
    catalog 仍然非常大，呼叫端應先用 priority 或 batch 管理工作量。
    """
    if card_count not in {3, 4, 5}:
        raise ValueError("card_count must be 3, 4, or 5")
    seen: set[tuple[str, ...]] = set()
    for board in combinations(FULL_DECK, card_count):
        canonical = canonicalize_board(board)
        if canonical not in seen:
            seen.add(canonical)
            yield canonical
            if limit is not None and len(seen) >= limit:
                return
