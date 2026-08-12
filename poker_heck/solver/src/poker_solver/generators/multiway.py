"""由規則引擎列舉 multiway postflop 訓練工作。"""
from __future__ import annotations
from argparse import ArgumentParser
import json
from itertools import islice, product
from pathlib import Path
from typing import Any
from poker_solver.engine.board_catalog import iter_canonical_boards
from poker_solver.engine.preflop_policy import PreflopSizingPolicy, abstract_actions
from poker_solver.engine.river_game import Action
from poker_solver.engine.table import apply_action, create_8max_preflop, is_terminal

def main() -> None:
    parser=ArgumentParser(description="產生 multiway 訓練工作"); parser.add_argument("grid",type=Path); args=parser.parse_args()
    path=args.grid.resolve(); raw=json.loads(path.read_text(encoding="utf-8")); output=_resolve(path.parent,raw["output_dir"]); manifest=_resolve(path.parent,raw["manifest"])
    jobs=build_jobs(raw,output); output.mkdir(parents=True,exist_ok=True)
    for job, config in jobs: (output/Path(job["config"]).name).write_text(json.dumps(config,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    manifest.parent.mkdir(parents=True,exist_ok=True); manifest.write_text(json.dumps({"strategy_db":raw["strategy_db"],"jobs":[j for j,_ in jobs]},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"已產生 {len(jobs)} 個 multiway pack：{output}")

def build_jobs(raw: dict[str,Any], output: Path) -> list[tuple[dict[str,Any],dict[str,Any]]]:
    if raw["solve_scope"] != "flop_full_tree":
        raise ValueError("multiway grid solve_scope must be flop_full_tree; turn/river use a conditional subgame config")
    stacks=tuple(int(x) for x in raw["stack_bb"]); counts=set(raw["player_counts"]); boards=_boards(raw); policy=PreflopSizingPolicy(**raw["preflop_route_policy"])
    routes=[(stack,route) for stack in stacks for route in _routes(stack,policy,counts,raw["max_preflop_routes_per_stack"])]
    keys=[(stack,route_id,actions,board_id,board) for stack,(route_id,count,actions) in routes for board_id,board in boards.items()]
    jobs=[]
    for seed,(stack,route_id,actions,board_id,board) in enumerate(keys,start=int(raw["seed_start"])):
        stem=f"mw_{seed:06d}_{route_id}_{board_id}_{stack}bb"
        config={"solve_scope":"flop_full_tree","stack_bb":stack,"board":list(board),"completed_street_actions":[],"preflop_actions":actions,"range_spec":raw["range_spec"],"sizing_policy":raw["sizing_policy"],"seed":seed}
        opt=raw["job_options"]
        jobs.append(({"game_type":"multiway_postflop","config":f"{output.name}/{stem}.json","range_profile_id":f"{raw['range_spec']['kind']}-{route_id}-{board_id}-{stack}bb","solver_version":raw["solver_version"],"iterations":int(raw["iterations_per_pack"]),"checkpoint":f"{raw['checkpoint_dir']}/{stem}.pkl","checkpoint_every":int(raw["checkpoint_every"]),"export_all_routes":bool(opt["export_all_routes"]),"quality_report":bool(opt["quality_report"])},config))
    return jobs

def _routes(stack:int, policy:PreflopSizingPolicy, counts:set[int], limit:int|None) -> list[tuple[str,int,list[dict[str,Any]]]]:
    result=[]
    def walk(state):
        if limit is not None and len(result) >= limit: return
        if is_terminal(state):
            if not state.hand_ended:
                active=sum(not p.folded for p in state.players); actions=[_action(a) for a in state.action_history]; result.append((f"p{active}_{len(result)+1:06d}",active,actions))
            return
        for action in abstract_actions(state,policy): walk(apply_action(state,action))
    walk(create_8max_preflop(stack_bb=stack)); return result

def _action(action:Action)->dict[str,Any]:
    item={"kind":action.kind.value}
    if action.amount is not None: item["amount_bb"]=action.amount/100
    return item

def _boards(raw:dict[str,Any])->dict[str,tuple[str,...]]:
    if raw["board_source"]=="explicit": return {k:tuple(v) for k,v in raw["board_buckets"].items()}
    cards=iter_canonical_boards(3,limit=raw["max_canonical_boards"])
    return {f"canonical_{i:06d}":b for i,b in enumerate(cards,1)}

def _resolve(base:Path,value:str)->Path:
    p=Path(value); return p if p.is_absolute() else (base/p).resolve()
if __name__=="__main__": main()
