"""預訓練策略的 SQLite 儲存與精確查詢。

第一版先支援 heads-up river。資料庫只回傳已離線訓練並寫入的策略，
未命中時由呼叫端決定要使用最近的 bucket 或排入背景 re-solve。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator

from poker_solver.engine.money import bb_to_units
from poker_solver.engine.postflop_game import PostflopGameState
from poker_solver.engine.river_game import Action, ActionType, Player, RiverGameState, create_river_game, format_action_as_pot_ratio
from poker_solver.engine.preflop_policy import format_action as format_preflop_action
from poker_solver.engine.table import MultiwayPostflopState, Position, PreflopState
from poker_solver.engine.multiway_postflop_policy import format_multiway_postflop_action_as_pot_ratio
from poker_solver.solver_core.multiway_postflop_mccfr import MultiwayPostflopMCCFRTrainer
from poker_solver.solver_core.preflop_mccfr import MultiwayPreflopMCCFRTrainer
from poker_solver.solver_core.river_analysis import RiverQualityReport
from poker_solver.solver_core.river_mccfr import RiverMCCFRTrainer
from poker_solver.solver_core.turn_mccfr import FlopMCCFRTrainer, TurnMCCFRTrainer


@dataclass(frozen=True)
class StoredAction:
    kind: str
    amount_units: int | None
    display: str
    probability: float
    samples: int | None = None
    ev_mean: float | None = None
    ev_stddev: float | None = None
    ev_stderr: float | None = None
    ci95_low: float | None = None
    ci95_high: float | None = None


def stored_action_with_stats(action: Action, display: str, probability: float, node: Any) -> StoredAction:
    stats = node.action_value_stats(action)
    return StoredAction(action.kind.value, action.amount, display, probability, **stats)


@dataclass(frozen=True)
class StoredRiverStrategy:
    strategy_key: str
    board: tuple[str, str, str, str, str]
    hero_position: str
    hero_cards: tuple[str, str]
    pot_units: int
    effective_stack_units: int
    action_history: tuple[tuple[str, int | None], ...]
    range_profile_id: str
    solver_version: str
    trained_iterations: int
    actions: tuple[StoredAction, ...]
    quality: dict[str, Any] | None


@dataclass(frozen=True)
class StrategyContext:
    """跨 street 的策略 lookup context；金額一律使用內部 units。"""

    game_type: str
    street: str
    player_count: int
    hero_position: str
    hero_cards: tuple[str, str]
    board: tuple[str, ...]
    pot_units: int
    effective_stack_units: int
    action_history: tuple[tuple[str, int | None], ...]
    range_profile_id: str
    solver_version: str


@dataclass(frozen=True)
class StoredStrategy:
    strategy_key: str
    context: StrategyContext
    trained_iterations: int
    actions: tuple[StoredAction, ...]
    quality: dict[str, Any] | None


@dataclass(frozen=True)
class ExportReport:
    reachable_infosets: int
    stored_infosets: int
    unvisited_infosets: int


def strategy_key(context: StrategyContext) -> str:
    payload = asdict(context)
    payload["hero_cards"] = tuple(sorted(context.hero_cards))
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def river_strategy_key(
    *,
    board: tuple[str, str, str, str, str],
    hero_position: Player | str,
    hero_cards: tuple[str, str],
    pot_units: int,
    effective_stack_units: int,
    action_history: tuple[tuple[str, int | None], ...] = (),
    range_profile_id: str,
    solver_version: str,
) -> str:
    """建立不受 JSON 欄位順序影響的精確 river lookup key。"""
    payload = {
        "board": board,
        "hero_position": hero_position.value if isinstance(hero_position, Player) else hero_position,
        "hero_cards": tuple(sorted(hero_cards)),
        "pot_units": pot_units,
        "effective_stack_units": effective_stack_units,
        "action_history": action_history,
        "range_profile_id": range_profile_id,
        "solver_version": solver_version,
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


class StrategyStore:
    """SQLite 策略庫；每個 context key 僅保留最新一次寫入的策略。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._batch_connection: sqlite3.Connection | None = None
        self._pending_strategies: list[StoredStrategy] | None = None
        self._pending_batch_size = 0
        self._pending_progress: Any | None = None
        self._create_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _create_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS river_strategies (
                    strategy_key TEXT PRIMARY KEY,
                    board_json TEXT NOT NULL,
                    hero_position TEXT NOT NULL,
                    hero_cards_json TEXT NOT NULL,
                    pot_units INTEGER NOT NULL,
                    effective_stack_units INTEGER NOT NULL,
                    action_history_json TEXT NOT NULL,
                    range_profile_id TEXT NOT NULL,
                    solver_version TEXT NOT NULL,
                    trained_iterations INTEGER NOT NULL,
                    actions_json TEXT NOT NULL,
                    quality_json TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS strategies (
                    strategy_key TEXT PRIMARY KEY,
                    game_type TEXT NOT NULL,
                    street TEXT NOT NULL,
                    player_count INTEGER NOT NULL,
                    hero_position TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    trained_iterations INTEGER NOT NULL,
                    actions_json TEXT NOT NULL,
                    quality_json TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """CREATE INDEX IF NOT EXISTS river_lookup_index ON river_strategies
                (range_profile_id, solver_version, hero_position)"""
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS completed_pack_exports (
                    export_key TEXT PRIMARY KEY,
                    trained_iterations INTEGER NOT NULL,
                    infoset_count INTEGER NOT NULL,
                    completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    @contextmanager
    def batch(self) -> Iterator[None]:
        """在單一 SQLite 交易中寫入大量策略，避免每個 infoset 各 commit 一次。"""
        if self._batch_connection is not None:
            yield
            return
        with self._connect() as connection:
            self._batch_connection = connection
            try:
                yield
            finally:
                self._batch_connection = None

    def upsert(self, strategy: StoredStrategy) -> None:
        if self._pending_strategies is not None:
            self._pending_strategies.append(strategy)
            if len(self._pending_strategies) >= self._pending_batch_size:
                self._flush_pending_strategies()
            return
        if self._batch_connection is not None:
            self._upsert(self._batch_connection, strategy)
            return
        with self._connect() as connection:
            self._upsert(connection, strategy)

    @contextmanager
    def buffered_upserts(self, *, batch_size: int = 500, progress: Any | None = None) -> Iterator[None]:
        """累積策略後以 executemany 寫入，降低大量 infoset 的 SQLite 呼叫成本。"""
        if batch_size <= 0:
            raise ValueError("batch_size 必須是正整數")
        if self._pending_strategies is not None:
            yield
            return
        self._pending_strategies = []
        self._pending_batch_size = batch_size
        self._pending_progress = progress
        try:
            yield
        except BaseException:
            self._pending_strategies.clear()
            raise
        else:
            self._flush_pending_strategies()
        finally:
            self._pending_strategies = None
            self._pending_batch_size = 0
            self._pending_progress = None

    def _flush_pending_strategies(self) -> None:
        strategies = self._pending_strategies
        if not strategies:
            return
        if self._batch_connection is not None:
            self._upsert_many(self._batch_connection, strategies)
        else:
            with self._connect() as connection:
                self._upsert_many(connection, strategies)
        if self._pending_progress is not None:
            self._pending_progress.update(len(strategies))
        strategies.clear()

    def _upsert(self, connection: sqlite3.Connection, strategy: StoredStrategy) -> None:
        self._upsert_many(connection, (strategy,))

    def _upsert_many(self, connection: sqlite3.Connection, strategies: tuple[StoredStrategy, ...] | list[StoredStrategy]) -> None:
        connection.executemany(
            """
            INSERT INTO strategies (
                strategy_key, game_type, street, player_count, hero_position,
                context_json, trained_iterations, actions_json, quality_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(strategy_key) DO UPDATE SET
                trained_iterations=excluded.trained_iterations,
                actions_json=excluded.actions_json,
                quality_json=excluded.quality_json,
                created_at=CURRENT_TIMESTAMP
            """,
            (
                (
                    strategy.strategy_key,
                    strategy.context.game_type,
                    strategy.context.street,
                    strategy.context.player_count,
                    strategy.context.hero_position,
                    json.dumps(asdict(strategy.context), sort_keys=True),
                    strategy.trained_iterations,
                    json.dumps([asdict(action) for action in strategy.actions]),
                    json.dumps(strategy.quality) if strategy.quality is not None else None,
                )
                for strategy in strategies
            ),
        )

    def is_pack_export_complete(self, export_key: str, *, trained_iterations: int, infoset_count: int) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM completed_pack_exports WHERE export_key = ? AND trained_iterations = ? AND infoset_count = ?",
                (export_key, trained_iterations, infoset_count),
            ).fetchone()
        return row is not None

    def mark_pack_export_complete(self, export_key: str, *, trained_iterations: int, infoset_count: int) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO completed_pack_exports (export_key, trained_iterations, infoset_count)
                VALUES (?, ?, ?)
                ON CONFLICT(export_key) DO UPDATE SET
                    trained_iterations=excluded.trained_iterations,
                    infoset_count=excluded.infoset_count,
                    completed_at=CURRENT_TIMESTAMP
                """,
                (export_key, trained_iterations, infoset_count),
            )

    def lookup(self, context: StrategyContext) -> StoredStrategy | None:
        key = strategy_key(context)
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM strategies WHERE strategy_key = ?", (key,)).fetchone()
        if row is None:
            return None
        raw_context = json.loads(row["context_json"])
        return StoredStrategy(
            strategy_key=row["strategy_key"],
            context=StrategyContext(
                game_type=raw_context["game_type"], street=raw_context["street"], player_count=raw_context["player_count"],
                hero_position=raw_context["hero_position"], hero_cards=tuple(raw_context["hero_cards"]), board=tuple(raw_context["board"]),
                pot_units=raw_context["pot_units"], effective_stack_units=raw_context["effective_stack_units"],
                action_history=tuple(tuple(item) for item in raw_context["action_history"]),
                range_profile_id=raw_context["range_profile_id"], solver_version=raw_context["solver_version"],
            ),
            trained_iterations=row["trained_iterations"],
            actions=tuple(StoredAction(**item) for item in json.loads(row["actions_json"])),
            quality=json.loads(row["quality_json"]) if row["quality_json"] else None,
        )

    def upsert_river(self, strategy: StoredRiverStrategy) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO river_strategies (
                    strategy_key, board_json, hero_position, hero_cards_json, pot_units,
                    effective_stack_units, action_history_json, range_profile_id, solver_version,
                    trained_iterations, actions_json, quality_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(strategy_key) DO UPDATE SET
                    trained_iterations=excluded.trained_iterations,
                    actions_json=excluded.actions_json,
                    quality_json=excluded.quality_json,
                    created_at=CURRENT_TIMESTAMP
                """,
                (
                    strategy.strategy_key,
                    json.dumps(strategy.board),
                    strategy.hero_position,
                    json.dumps(strategy.hero_cards),
                    strategy.pot_units,
                    strategy.effective_stack_units,
                    json.dumps(strategy.action_history),
                    strategy.range_profile_id,
                    strategy.solver_version,
                    strategy.trained_iterations,
                    json.dumps([asdict(action) for action in strategy.actions]),
                    json.dumps(strategy.quality) if strategy.quality is not None else None,
                ),
            )

    def lookup_river(
        self,
        *,
        board: tuple[str, str, str, str, str],
        hero_position: Player | str,
        hero_cards: tuple[str, str],
        pot_bb: int | float | str,
        effective_stack_bb: int | float | str,
        range_profile_id: str,
        solver_version: str = "river-v1",
    ) -> StoredRiverStrategy | None:
        pot_units = bb_to_units(pot_bb)
        stack_units = bb_to_units(effective_stack_bb)
        key = river_strategy_key(
            board=board,
            hero_position=hero_position,
            hero_cards=hero_cards,
            pot_units=pot_units,
            effective_stack_units=stack_units,
            range_profile_id=range_profile_id,
            solver_version=solver_version,
        )
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM river_strategies WHERE strategy_key = ?", (key,)).fetchone()
        return _row_to_strategy(row) if row is not None else None


def store_river_root_strategy(
    store: StrategyStore,
    trainer: RiverMCCFRTrainer,
    state: RiverGameState,
    player: Player,
    *,
    range_profile_id: str = "default",
    solver_version: str = "river-v1",
    quality: RiverQualityReport | None = None,
) -> StoredRiverStrategy:
    """將已訓練 river 根節點的平均策略寫入策略庫。"""
    hero_cards = state.player_state(player).hole_cards
    history = tuple((action.kind.value, action.amount) for action in state.action_history)
    key = river_strategy_key(
        board=state.board,
        hero_position=player,
        hero_cards=hero_cards,
        pot_units=state.pot,
        effective_stack_units=bb_to_units(trainer.effective_stack_bb),
        action_history=history,
        range_profile_id=range_profile_id,
        solver_version=solver_version,
    )
    strategy = trainer.strategy_for(state, player)
    stored = StoredRiverStrategy(
        strategy_key=key,
        board=state.board,
        hero_position=player.value,
        hero_cards=tuple(sorted(hero_cards)),
        pot_units=state.pot,
        effective_stack_units=bb_to_units(trainer.effective_stack_bb),
        action_history=history,
        range_profile_id=range_profile_id,
        solver_version=solver_version,
        trained_iterations=trainer.iterations_completed,
        actions=tuple(
            StoredAction(action.kind.value, action.amount, format_action_as_pot_ratio(state, action), probability)
            for action, probability in strategy.items()
        ),
        quality=asdict(quality) if quality is not None else None,
    )
    store.upsert_river(stored)
    store.upsert(
        StoredStrategy(
            strategy_key=strategy_key(
                StrategyContext(
                    game_type="river_heads_up", street="river", player_count=2, hero_position=player.value,
                    hero_cards=tuple(sorted(hero_cards)), board=state.board, pot_units=state.pot,
                    effective_stack_units=bb_to_units(trainer.effective_stack_bb), action_history=history,
                    range_profile_id=range_profile_id, solver_version=solver_version,
                )
            ),
            context=StrategyContext(
                game_type="river_heads_up", street="river", player_count=2, hero_position=player.value,
                hero_cards=tuple(sorted(hero_cards)), board=state.board, pot_units=state.pot,
                effective_stack_units=bb_to_units(trainer.effective_stack_bb), action_history=history,
                range_profile_id=range_profile_id, solver_version=solver_version,
            ),
            trained_iterations=trainer.iterations_completed,
            actions=stored.actions,
            quality=stored.quality,
        )
    )
    return stored


def store_river_state_strategy(  # pragma: no cover - replaced by direct visited-infoset export
    store: StrategyStore,
    trainer: RiverMCCFRTrainer,
    state: RiverGameState,
    player: Player,
    *,
    range_profile_id: str,
    solver_version: str,
    quality: RiverQualityReport | None = None,
) -> StoredStrategy:
    """匯出 river tree 任一已訓練資訊集至通用策略表。"""
    hero_cards = tuple(sorted(state.player_state(player).hole_cards))
    history = tuple((action.kind.value, action.amount) for action in state.action_history)
    context = StrategyContext(
        game_type="river_heads_up", street="river", player_count=2, hero_position=player.value,
        hero_cards=hero_cards, board=state.board, pot_units=state.pot,
        effective_stack_units=bb_to_units(trainer.effective_stack_bb), action_history=history,
        range_profile_id=range_profile_id, solver_version=solver_version,
    )
    record = StoredStrategy(
        strategy_key=strategy_key(context), context=context, trained_iterations=trainer.iterations_completed,
        actions=tuple(
            StoredAction(action.kind.value, action.amount, format_action_as_pot_ratio(state, action), probability)
            for action, probability in trainer.strategy_for(state, player).items()
        ),
        quality=asdict(quality) if quality is not None else None,
    )
    store.upsert(record)
    return record


def export_river_tree(
    store: StrategyStore,
    trainer: RiverMCCFRTrainer,
    *,
    range_profile_id: str = "default",
    solver_version: str = "river-v1",
    quality: RiverQualityReport | None = None,
) -> ExportReport:
    """遍歷固定 river 子遊戲所有抽象路徑，匯出每個已訓練資訊集。

    未曾被 external sampling 走訪的節點不會捏造 uniform 策略，而會計入
    ``unvisited_infosets``，讓夜間 pipeline 能以覆蓋率決定是否再訓練。
    """
    """只匯出 MCCFR 已訪問的 river infoset，避免全 range 交叉窮舉。"""
    stored = 0
    for key, node in trainer.infosets.items():
        position, hole_cards, board, pot, _current_bet, _raise_size, history = key
        context = StrategyContext(
            game_type="river_heads_up", street="river", player_count=2, hero_position=str(position),
            hero_cards=tuple(sorted(hole_cards)), board=tuple(board), pot_units=int(pot),
            effective_stack_units=bb_to_units(trainer.effective_stack_bb), action_history=tuple(history),
            range_profile_id=range_profile_id, solver_version=solver_version,
        )
        record = StoredStrategy(
            strategy_key=strategy_key(context), context=context, trained_iterations=trainer.iterations_completed,
            actions=tuple(stored_action_with_stats(action, _format_river_action(action, int(pot)), probability, node) for action, probability in node.average_strategy().items()),
            quality=asdict(quality) if quality is not None else None,
        )
        store.upsert(record)
        stored += 1
    return ExportReport(stored, stored, 0)


def export_preflop_infosets(
    store: StrategyStore,
    trainer: MultiwayPreflopMCCFRTrainer,
    *,
    range_profile_id: str = "default",
    solver_version: str = "preflop-v1",
) -> ExportReport:
    """匯出已訓練到的所有 8-Max preflop infosets。"""
    stored = 0
    for key, node in trainer.infosets.items():
        position, hole_cards, pot, _current_bet, _raise_size, _seats, history = key
        context = StrategyContext(
            game_type="preflop_8max", street="preflop", player_count=8, hero_position=str(position),
            hero_cards=tuple(sorted(hole_cards)), board=(), pot_units=int(pot), effective_stack_units=bb_to_units(trainer.stack_bb),
            action_history=tuple(history), range_profile_id=range_profile_id, solver_version=solver_version,
        )
        record = StoredStrategy(
            strategy_key=strategy_key(context), context=context, trained_iterations=trainer.iterations_completed,
            actions=tuple(
                stored_action_with_stats(action, _format_preflop_action(action), probability, node)
                for action, probability in node.average_strategy().items()
            ),
            quality=None,
        )
        store.upsert(record)
        stored += 1
    return ExportReport(stored, stored, 0)


def export_multiway_postflop_infosets(
    store: StrategyStore,
    trainer: MultiwayPostflopMCCFRTrainer,
    *,
    range_profile_id: str = "default",
    solver_version: str = "multiway-postflop-v1",
) -> ExportReport:
    """匯出已訓練到的多人 postflop infosets。"""
    stored = 0
    default_stack = max(player.stack for player in trainer.initial_state.players)
    for key, node in trainer.infosets.items():
        position, hole_cards, street, board, pot, current_bet, _raise_size, seats, _pending, _raise_allowed, history = key
        committed = next(seat[2] for seat in seats if seat[0] == position)
        to_call = current_bet - committed
        context = StrategyContext(
            game_type="multiway_postflop", street=str(street), player_count=sum(not seat[4] for seat in seats), hero_position=str(position),
            hero_cards=tuple(sorted(hole_cards)), board=tuple(board), pot_units=int(pot), effective_stack_units=default_stack,
            action_history=tuple(history), range_profile_id=range_profile_id, solver_version=solver_version,
        )
        record = StoredStrategy(
            strategy_key=strategy_key(context), context=context, trained_iterations=trainer.iterations_completed,
            actions=tuple(
                stored_action_with_stats(action, _format_multiway_action(action, int(pot), int(current_bet), int(to_call)), probability, node)
                for action, probability in node.average_strategy().items()
            ),
            quality=None,
        )
        store.upsert(record)
        stored += 1
    return ExportReport(stored, stored, 0)


def export_multiway_postflop_root_infosets(
    store: StrategyStore,
    trainer: MultiwayPostflopMCCFRTrainer,
    *,
    range_profile_id: str = "default",
    solver_version: str = "multiway-postflop-v1",
) -> ExportReport:
    """只保存起始決策點；完整子樹保留在 checkpoint 供後續 re-solve。"""
    state = trainer.initial_state
    actor = state.current_player
    if actor is None:
        return ExportReport(0, 0, 0)
    stored = 0
    default_stack = max(player.stack for player in state.players)
    for key, node in trainer.infosets.items():
        position, hole_cards, street, board, pot, current_bet, _raise_size, seats, _pending, _raise_allowed, history = key
        if position != actor.value or tuple(board) != state.board or tuple(history) != ():
            continue
        committed = next(seat[2] for seat in seats if seat[0] == position)
        to_call = current_bet - committed
        context = StrategyContext(
            game_type="multiway_postflop", street=str(street), player_count=sum(not seat[4] for seat in seats), hero_position=str(position),
            hero_cards=tuple(sorted(hole_cards)), board=tuple(board), pot_units=int(pot), effective_stack_units=default_stack,
            action_history=(), range_profile_id=range_profile_id, solver_version=solver_version,
        )
        record = StoredStrategy(
            strategy_key=strategy_key(context), context=context, trained_iterations=trainer.iterations_completed,
            actions=tuple(
                stored_action_with_stats(action, _format_multiway_action(action, int(pot), int(current_bet), int(to_call)), probability, node)
                for action, probability in node.average_strategy().items()
            ),
            quality={"storage_scope": "root_only", "total_trained_infosets": len(trainer.infosets)},
        )
        store.upsert(record)
        stored += 1
    return ExportReport(stored, stored, 0)


def export_heads_up_postflop_infosets(
    store: StrategyStore,
    trainer: FlopMCCFRTrainer | TurnMCCFRTrainer,
    *,
    game_type: str,
    range_profile_id: str = "default",
    solver_version: str = "heads-up-postflop-v1",
) -> ExportReport:
    """匯出已訓練的單挑 flop／turn／river 資訊集。

    Flop 與 turn trainer 含有 chance node，有限 iteration 不會走遍所有後續
    公共牌；此函式只匯出實際訓練過的資訊集，不以假資料補齊。
    """
    stored = 0
    for key, node in trainer.infosets.items():
        street, player, hole_cards, board, pot, current_bet, _raise_size, history = key
        context = StrategyContext(
            game_type=game_type, street=str(street), player_count=2, hero_position=str(player),
            hero_cards=tuple(sorted(hole_cards)), board=tuple(board), pot_units=int(pot),
            effective_stack_units=bb_to_units(trainer.effective_stack_bb), action_history=tuple(history),
            range_profile_id=range_profile_id, solver_version=solver_version,
        )
        record = StoredStrategy(
            strategy_key=strategy_key(context), context=context, trained_iterations=trainer.iterations_completed,
            actions=tuple(
                stored_action_with_stats(action, _format_heads_up_postflop_action(action, int(pot)), probability, node)
                for action, probability in node.average_strategy().items()
            ), quality=None,
        )
        store.upsert(record)
        stored += 1
    return ExportReport(stored, stored, 0)


def store_heads_up_postflop_root_strategy(
    store: StrategyStore,
    trainer: FlopMCCFRTrainer | TurnMCCFRTrainer,
    state: PostflopGameState,
    player: Player,
    *,
    game_type: str,
    range_profile_id: str = "default",
    solver_version: str = "heads-up-postflop-v1",
) -> StoredStrategy:
    """將單挑 flop 或 turn 起始節點的平均策略寫入通用資料庫。"""
    context = StrategyContext(
        game_type=game_type, street=state.street.value, player_count=2, hero_position=player.value,
        hero_cards=tuple(sorted(state.player_state(player).hole_cards)), board=state.board,
        pot_units=state.pot, effective_stack_units=bb_to_units(trainer.effective_stack_bb),
        action_history=(), range_profile_id=range_profile_id, solver_version=solver_version,
    )
    record = StoredStrategy(
        strategy_key=strategy_key(context), context=context, trained_iterations=trainer.iterations_completed,
        actions=tuple(
            StoredAction(action.kind.value, action.amount, _format_heads_up_postflop_action(action, state.pot), probability)
            for action, probability in trainer.strategy_for(state, player).items()
        ), quality=None,
    )
    store.upsert(record)
    return record


def _format_preflop_action(action: Action) -> str:
    if action.amount is None:
        return action.kind.value.replace("_", "-")
    from poker_solver.engine.money import format_bb

    return f"raise to {format_bb(action.amount)}"


def _format_river_action(action: Action, pot: int) -> str:
    if action.kind in {ActionType.FOLD, ActionType.CHECK, ActionType.CALL}:
        return action.kind.value
    if action.kind is ActionType.ALL_IN:
        return "all-in"
    assert action.amount is not None
    return f"{action.kind.value} {100 * action.amount / pot:g}% pot"


def _format_multiway_action(action: Action, pot: int, current_bet: int, to_call: int) -> str:
    if action.kind in {ActionType.FOLD, ActionType.CHECK, ActionType.CALL}:
        return action.kind.value
    if action.kind is ActionType.ALL_IN:
        return "all-in"
    assert action.amount is not None
    if action.kind is ActionType.BET:
        return f"bet {100 * action.amount / pot:g}% pot"
    return f"raise {100 * (action.amount - current_bet) / (pot + to_call):g}% pot-after-call"


def _format_heads_up_postflop_action(action: Action, pot: int) -> str:
    if action.kind in {ActionType.FOLD, ActionType.CHECK, ActionType.CALL}:
        return action.kind.value
    if action.kind is ActionType.ALL_IN:
        return "all-in"
    assert action.amount is not None
    if action.kind is ActionType.BET:
        return f"bet {100 * action.amount / pot:g}% pot"
    return f"raise to {action.amount} units"


def store_preflop_root_strategy(
    store: StrategyStore,
    trainer: MultiwayPreflopMCCFRTrainer,
    state: PreflopState,
    player: Position,
    hole_cards: tuple[str, str],
    *,
    range_profile_id: str = "default",
    solver_version: str = "preflop-v1",
) -> StoredStrategy:
    """將 8-Max preflop 根節點策略寫入通用策略庫。"""
    context = StrategyContext(
        game_type="preflop_8max", street="preflop", player_count=8, hero_position=player.value,
        hero_cards=tuple(sorted(hole_cards)), board=(), pot_units=state.pot,
        effective_stack_units=bb_to_units(trainer.stack_bb), action_history=(),
        range_profile_id=range_profile_id, solver_version=solver_version,
    )
    record = StoredStrategy(
        strategy_key=strategy_key(context), context=context, trained_iterations=trainer.iterations_completed,
        actions=tuple(
            StoredAction(action.kind.value, action.amount, format_preflop_action(state, action), probability)
            for action, probability in trainer.strategy_for(state, player, hole_cards).items()
        ),
        quality=None,
    )
    store.upsert(record)
    return record


def store_multiway_postflop_root_strategy(
    store: StrategyStore,
    trainer: MultiwayPostflopMCCFRTrainer,
    state: MultiwayPostflopState,
    player: Position,
    hole_cards: tuple[str, str],
    *,
    range_profile_id: str = "default",
    solver_version: str = "multiway-postflop-v1",
) -> StoredStrategy:
    """將多人 postflop 根節點策略寫入通用策略庫。"""
    context = StrategyContext(
        game_type="multiway_postflop", street=state.street, player_count=len(state.players), hero_position=player.value,
        hero_cards=tuple(sorted(hole_cards)), board=state.board, pot_units=state.pot,
        effective_stack_units=max(player_state.stack for player_state in state.players), action_history=(),
        range_profile_id=range_profile_id, solver_version=solver_version,
    )
    record = StoredStrategy(
        strategy_key=strategy_key(context), context=context, trained_iterations=trainer.iterations_completed,
        actions=tuple(
            StoredAction(action.kind.value, action.amount, format_multiway_postflop_action_as_pot_ratio(state, action), probability)
            for action, probability in trainer.strategy_for(state, player, hole_cards).items()
        ),
        quality=None,
    )
    store.upsert(record)
    return record


def _row_to_strategy(row: sqlite3.Row) -> StoredRiverStrategy:
    return StoredRiverStrategy(
        strategy_key=row["strategy_key"],
        board=tuple(json.loads(row["board_json"])),  # type: ignore[arg-type]
        hero_position=row["hero_position"],
        hero_cards=tuple(json.loads(row["hero_cards_json"])),  # type: ignore[arg-type]
        pot_units=row["pot_units"],
        effective_stack_units=row["effective_stack_units"],
        action_history=tuple(tuple(item) for item in json.loads(row["action_history_json"])),
        range_profile_id=row["range_profile_id"],
        solver_version=row["solver_version"],
        trained_iterations=row["trained_iterations"],
        actions=tuple(StoredAction(**item) for item in json.loads(row["actions_json"])),
        quality=json.loads(row["quality_json"]) if row["quality_json"] else None,
    )
