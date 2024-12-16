import numpy as np
import argparse
import json
import os
import yaml
from evaluation.util import action_to_config, get_cutting_planes, get_model_files
from util import CONFIGS_DIR, SEARCH_HISTORY_DIR
from evaluation.solve import chunked_evaluate


def extract_action(file_path: str) -> np.array:
    with open(file_path, 'rb') as f:
        return json.load(f)['action']

def build_baseline_configs():
    for family in os.listdir(SEARCH_HISTORY_DIR):
        for solver_folder in os.listdir(f"{SEARCH_HISTORY_DIR}{family}"):
            outputs = os.listdir(f"{SEARCH_HISTORY_DIR}{family}/{solver_folder}")
            outputs.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
            best_path = f"{SEARCH_HISTORY_DIR}{family}/{solver_folder}/{outputs[-1]}"
            action = extract_action(best_path)
            solver = solver_folder.split('_')[-1]
            config = action_to_config(action, solver)
            write_path = f"{CONFIGS_DIR}/{family}/{solver_folder}/random_search/"
            if not os.path.exists(write_path):
                os.makedirs(write_path)
            with open(f"{write_path}baseline.yaml", 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

def _actions_fn(args: argparse.Namespace) -> list[np.array]:
    rng = np.random.default_rng(args.seed)
    match args.solver:
        case "gurobi":
            return [
                rng.integers(low=0, high=3, size=len(get_cutting_planes("gurobi")))
                for _ in range(args.num_configs)
            ]

        case "scip":
            return [
                rng.integers(low=0, high=2, size=len(get_cutting_planes("scip")))
                for _ in range(args.num_configs)
            ]

        case _:
            raise NotImplementedError(f"Solver {args.solver} is not supported right now...")

def _record_fn(args: argparse.Namespace, best_action: np.array, best_improv: float, iteration: int) -> None:
    write_path = os.path.join(SEARCH_HISTORY_DIR, args.instance_name, args.solver)
    os.makedirs(write_path, exist_ok=True)
    results = {
        "action": best_action.tolist(),
        "improvement": best_improv,
        "num_solves": args.num_solves
    }
    with open(f"{write_path}/output_{iteration}.txt", "w") as file:
        file.write(json.dumps(results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data input
    parser.add_argument('--instance_name', type=str, help="the MILP class")

    # Exploration settings
    parser.add_argument('--num_configs', type=int, default=1000, help="number of configurations to sample")
    parser.add_argument('--chunk_size', type=int, default=100000, help="max number of actions to evaluate in parallel")

    # Solver settings
    parser.add_argument('--solver', type=str, default="gurobi", help="which MILP solver to use")
    parser.add_argument('--num_solves', type=int, default=10, help="number of runtimes to average on each instance")
    parser.add_argument('--gap_limit', type=float, default=0.0, help="gap limit for solving instances")
    parser.add_argument('--time_limit', type=float, default=100.0, help="time limit for solving instances")
    
    args = parser.parse_args()
    args.seed = np.random.randint(0, 1000)
    args.eval_type = 'val'
    actions = _actions_fn(args)
    model_files = get_model_files(args)

    chunked_evaluate(
        args=args,
        actions=actions,
        model_files=model_files,
        record_fn=_record_fn
    )

    build_baseline_configs()


