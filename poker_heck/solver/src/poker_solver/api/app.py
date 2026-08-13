"""River solver 的 FastAPI 查詢服務。"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from poker_solver.engine.money import bb_to_units, format_bb
from poker_solver.engine.multiway_postflop_policy import format_multiway_postflop_action_as_pot_ratio
from poker_solver.engine.river_game import Action, ActionType, Player, create_river_game, format_action_as_pot_ratio
from poker_solver.engine.table import Position, apply_multiway_postflop_action
from poker_solver.solver_core.multiway_postflop_mccfr import MultiwayPostflopMCCFRTrainer
from poker_solver.solver_core.river_mccfr import RiverMCCFRTrainer
from poker_solver.solver_core.strategy_store import StrategyContext, StrategyStore


class RiverStrategyRequest(BaseModel):
    board: tuple[str, str, str, str, str]
    hero_position: Player
    hero_cards: tuple[str, str]
    pot_bb: float = Field(gt=0)
    effective_stack_bb: float = Field(gt=0)

    @model_validator(mode="after")
    def cards_must_be_distinct(self) -> "RiverStrategyRequest":
        if len(set((*self.board, *self.hero_cards))) != 7:
            raise ValueError("board and hero cards must contain seven distinct cards")
        return self


class StrategyActionResponse(BaseModel):
    action: str
    probability: float
    samples: int | None = None
    ev_mean: float | None = None
    ev_stddev: float | None = None
    ev_stderr: float | None = None
    ci95_low: float | None = None
    ci95_high: float | None = None


class RiverStrategyResponse(BaseModel):
    position: Player
    board: tuple[str, str, str, str, str]
    hero_cards: tuple[str, str]
    pot: str
    actions: list[StrategyActionResponse]
    algorithm: str = "external_sampling_mccfr"
    iterations: int
    infosets: int


class RiverDatabaseStrategyRequest(RiverStrategyRequest):
    range_profile_id: str = "default"
    solver_version: str = "river-v1"


class RiverDatabaseStrategyResponse(BaseModel):
    strategy_key: str
    position: Player
    board: tuple[str, str, str, str, str]
    hero_cards: tuple[str, str]
    pot: str
    actions: list[StrategyActionResponse]
    trained_iterations: int
    quality: dict[str, object] | None


class DecisionHistoryAction(BaseModel):
    kind: ActionType
    amount_units: int | None = Field(default=None, ge=0)


class DecisionRequest(BaseModel):
    game_type: str
    street: str
    player_count: int = Field(ge=2, le=8)
    hero_position: str
    hero_cards: tuple[str, str]
    board: tuple[str, ...] = ()
    pot_bb: float = Field(gt=0)
    effective_stack_bb: float = Field(gt=0)
    action_history: tuple[DecisionHistoryAction, ...] = ()
    range_profile_id: str = "default"
    solver_version: str

    @model_validator(mode="after")
    def cards_must_be_distinct(self) -> "DecisionRequest":
        if len(set((*self.board, *self.hero_cards))) != len(self.board) + 2:
            raise ValueError("board and hero cards must be distinct")
        return self


class DecisionResponse(BaseModel):
    strategy_key: str
    game_type: str
    street: str
    actions: list[StrategyActionResponse]
    trained_iterations: int
    quality: dict[str, object] | None


class MultiwayRootStrategyRequest(BaseModel):
    hero_position: Position
    hero_cards: tuple[str, str]
    actions: tuple["MultiwayActionRequest", ...] = ()

    @model_validator(mode="after")
    def cards_must_be_distinct(self) -> "MultiwayRootStrategyRequest":
        if len(set(self.hero_cards)) != 2:
            raise ValueError("hero cards must be distinct")
        return self


class MultiwayActionRequest(BaseModel):
    kind: ActionType
    pot_ratio: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def sizing_matches_action(self) -> "MultiwayActionRequest":
        needs_ratio = self.kind in {ActionType.BET, ActionType.RAISE}
        if needs_ratio != (self.pot_ratio is not None):
            raise ValueError("pot_ratio is required only for bet and raise")
        return self


class MultiwayRootStrategyResponse(BaseModel):
    position: Position
    street: str
    board: tuple[str, ...]
    hero_cards: tuple[str, str]
    pot: str
    actions: list[StrategyActionResponse]
    algorithm: str = "external_sampling_mccfr"
    iterations: int
    infosets: int


def create_app(
    trainer: RiverMCCFRTrainer | MultiwayPostflopMCCFRTrainer,
    strategy_store: StrategyStore | None = None,
) -> FastAPI:
    app = FastAPI(title="Poker Solver API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, int | str]:
        solver_type = "river_heads_up" if isinstance(trainer, RiverMCCFRTrainer) else "multiway_postflop"
        return {
            "status": "ok",
            "solver_type": solver_type,
            "iterations": trainer.iterations_completed,
            "infosets": len(trainer.infosets),
        }

    @app.post("/v1/river/strategy", response_model=RiverStrategyResponse)
    def river_strategy(request: RiverStrategyRequest) -> RiverStrategyResponse:
        if not isinstance(trainer, RiverMCCFRTrainer):
            raise HTTPException(status_code=404, detail="loaded checkpoint is not a river solver")
        try:
            state = _build_query_state(trainer, request)
            strategy = trainer.strategy_for(state, request.hero_position)
        except (KeyError, ValueError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return RiverStrategyResponse(
            position=request.hero_position,
            board=request.board,
            hero_cards=request.hero_cards,
            pot=format_bb(state.pot),
            actions=[StrategyActionResponse(action=format_action_as_pot_ratio(state, action), probability=probability) for action, probability in strategy.items()],
            iterations=trainer.iterations_completed,
            infosets=len(trainer.infosets),
        )

    @app.post("/v1/river/database-strategy", response_model=RiverDatabaseStrategyResponse)
    def river_database_strategy(request: RiverDatabaseStrategyRequest) -> RiverDatabaseStrategyResponse:
        if strategy_store is None:
            raise HTTPException(status_code=503, detail="strategy database is not configured")
        stored = strategy_store.lookup_river(
            board=request.board,
            hero_position=request.hero_position,
            hero_cards=request.hero_cards,
            pot_bb=request.pot_bb,
            effective_stack_bb=request.effective_stack_bb,
            range_profile_id=request.range_profile_id,
            solver_version=request.solver_version,
        )
        if stored is None:
            raise HTTPException(status_code=404, detail="no exact pre-trained strategy matches this context")
        return RiverDatabaseStrategyResponse(
            strategy_key=stored.strategy_key,
            position=Player(stored.hero_position),
            board=stored.board,
            hero_cards=stored.hero_cards,
            pot=format_bb(stored.pot_units),
            actions=[StrategyActionResponse(action=action.display, probability=action.probability, samples=action.samples, ev_mean=action.ev_mean, ev_stddev=action.ev_stddev, ev_stderr=action.ev_stderr, ci95_low=action.ci95_low, ci95_high=action.ci95_high) for action in stored.actions],
            trained_iterations=stored.trained_iterations,
            quality=stored.quality,
        )

    @app.post("/v1/decision", response_model=DecisionResponse)
    def decision(request: DecisionRequest) -> DecisionResponse:
        if strategy_store is None:
            raise HTTPException(status_code=503, detail="strategy database is not configured")
        context = StrategyContext(
            game_type=request.game_type,
            street=request.street,
            player_count=request.player_count,
            hero_position=request.hero_position,
            hero_cards=tuple(sorted(request.hero_cards)),
            board=request.board,
            pot_units=bb_to_units(request.pot_bb),
            effective_stack_units=bb_to_units(request.effective_stack_bb),
            action_history=tuple((action.kind.value, action.amount_units) for action in request.action_history),
            range_profile_id=request.range_profile_id,
            solver_version=request.solver_version,
        )
        stored = strategy_store.lookup(context)
        if stored is None:
            raise HTTPException(status_code=404, detail="no exact pre-trained strategy matches this context")
        return DecisionResponse(
            strategy_key=stored.strategy_key,
            game_type=stored.context.game_type,
            street=stored.context.street,
            actions=[StrategyActionResponse(action=action.display, probability=action.probability, samples=action.samples, ev_mean=action.ev_mean, ev_stddev=action.ev_stddev, ev_stderr=action.ev_stderr, ci95_low=action.ci95_low, ci95_high=action.ci95_high) for action in stored.actions],
            trained_iterations=stored.trained_iterations,
            quality=stored.quality,
        )

    @app.post("/v1/multiway-postflop/root-strategy", response_model=MultiwayRootStrategyResponse)
    def multiway_postflop_root_strategy(request: MultiwayRootStrategyRequest) -> MultiwayRootStrategyResponse:
        if not isinstance(trainer, MultiwayPostflopMCCFRTrainer):
            raise HTTPException(status_code=404, detail="loaded checkpoint is not a multiway postflop solver")
        state = trainer.initial_state
        try:
            for requested_action in request.actions:
                action = _multiway_action_from_ratio(state, requested_action)
                state = apply_multiway_postflop_action(state, action)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        if request.hero_position is not state.current_player:
            raise HTTPException(status_code=422, detail="only the trained root acting position can be queried here")
        if set(request.hero_cards) & set(state.board):
            raise HTTPException(status_code=422, detail="hero cards overlap the board")
        try:
            strategy = trainer.strategy_for(state, request.hero_position, request.hero_cards)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return MultiwayRootStrategyResponse(
            position=request.hero_position,
            street=state.street,
            board=state.board,
            hero_cards=request.hero_cards,
            pot=format_bb(state.pot),
            actions=[
                StrategyActionResponse(action=format_multiway_postflop_action_as_pot_ratio(state, action), probability=probability)
                for action, probability in strategy.items()
            ],
            iterations=trainer.iterations_completed,
            infosets=len(trainer.infosets),
        )

    return app


def _multiway_action_from_ratio(state, request: MultiwayActionRequest) -> Action:
    if request.kind not in {ActionType.BET, ActionType.RAISE}:
        return Action(request.kind)
    assert request.pot_ratio is not None
    if state.current_player is None:
        raise ValueError("cannot apply an action to a completed street")
    if request.kind is ActionType.BET:
        from math import ceil

        return Action(ActionType.BET, ceil(state.pot * request.pot_ratio))
    from math import ceil

    actor = state.player(state.current_player)
    return Action(ActionType.RAISE, state.current_bet + ceil((state.pot + state.call_amount(actor.position)) * request.pot_ratio))


def _build_query_state(trainer: RiverMCCFRTrainer, request: RiverStrategyRequest):
    opponent_range = trainer.ip_range if request.hero_position is Player.OOP else trainer.oop_range
    used = set(request.board) | set(request.hero_cards)
    opponent_combo = next((combo.cards for combo in opponent_range.combos if not (set(combo.cards) & used)), None)
    if opponent_combo is None:
        raise ValueError("no compatible opponent combo exists in the trained range")
    if request.hero_position is Player.OOP:
        return create_river_game(request.board, request.hero_cards, opponent_combo, initial_pot_bb=request.pot_bb, effective_stack_bb=request.effective_stack_bb)
    return create_river_game(request.board, opponent_combo, request.hero_cards, initial_pot_bb=request.pot_bb, effective_stack_bb=request.effective_stack_bb)
