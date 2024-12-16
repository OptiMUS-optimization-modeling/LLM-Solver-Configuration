import gurobipy as gp
import numpy as np
import pyscipopt as pyopt
import sys
from argparse import Namespace
from itertools import product
from joblib import Parallel, delayed
from pkg_resources import resource_filename
from tqdm import tqdm
from typing import Optional
from evaluation.util import chunk, get_cutting_planes, get_cpu_count, build_inputs

def _gurobi_solve(input: tuple, args: dict, fathom_time: float) -> float:
    action, file, seed = input
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.setParam("LogToConsole", 0)
        env.start()
        with gp.read(file, env=env) as model:
            model.setParam("Seed", seed)
            model.setParam("Threads", 1)
            model.setParam("MIPGap", args['gap_limit'])

            match action:
                case None:
                    model.optimize()
                    
                case _:
                    for i, sep in enumerate(get_cutting_planes("gurobi")):
                        model.setParam(sep, action[i])
                        
                    model.setParam("WorkLimit", min(args["time_limit"], 2.5 * fathom_time))
                    model.optimize()
            work = model.Work
    #time.sleep(1)
    return work

def _scip_solve(input: tuple, args: dict, fathom_time: float) -> float:
    action, file, _ = input
    model = pyopt.Model()
    model.setParam("display/verblevel", 0)
    model.readProblem(file)
    model.readParams(resource_filename(__name__, "randomness_control.set"))
    model.setParam("limits/gap", args["gap_limit"])

    match action:
        case None:
            model.optimize()
        case _:
            for i, sep in enumerate(get_cutting_planes("scip")):
                on_or_off = 2 * (action[i] > 0.5) - 1
                model.setParam(f"separating/{sep}/freq", on_or_off)

            model.setParam("limits/time", 2.5 * min(args["time_limit"], fathom_time))
            model.optimize()

    return model.getSolvingTime()

def solve(input: tuple, args: dict, fathom_time: float) -> float:
    solve_fns = {
        "gurobi": _gurobi_solve,
        "scip": _scip_solve
    }
    return solve_fns[args["solver"]](input, args, fathom_time)

def _get_solver_args(args: Namespace) -> dict:
    return {
        "solver": args.solver,
        "time_limit": args.time_limit,
        "gap_limit": args.gap_limit
    }

def parallel_solve(
    args: Namespace,
    inputs: list,
    num_actions: int,
    num_models: int,
    num_samples: int,
    fathom_times: dict
) -> np.ndarray:
    solve_args = _get_solver_args(args)
    runtimes = np.array(
        list(
            tqdm(
                Parallel(return_as="generator", n_jobs=args.n_cpus)(
                    delayed(solve)(input, solve_args, fathom_times[input[1]]) for input in inputs
                ),
                total=len(inputs),
                miniters=len(inputs) // 5,
                maxinterval=np.inf,
                file=sys.stdout
            )
        )
    )
    return runtimes.reshape(num_actions, num_models, num_samples)



def _compute_improv(default_times: np.array, times: np.array) -> np.array:
    repeated_defaults = np.repeat(default_times.squeeze(0)[np.newaxis, :, :], times.shape[0], axis=0)
    return (repeated_defaults - times) / repeated_defaults
  


def evaluate(
    args: Namespace,
    actions: list[np.ndarray],
    model_files: list[str],
    fathoming: Optional[bool] = False
) -> tuple:
    
    # Generate input, initialize solve args
    samples = np.arange(args.num_solves)
    args.n_cpus = get_cpu_count(args.solver)
    print(f"Running on {args.n_cpus} CPUs...")
    default_times = parallel_solve(
        args=args,
        inputs=list(product([None], model_files, samples)),
        num_actions=1,
        num_models=len(model_files),
        num_samples=len(samples),
        fathom_times=dict(zip(model_files, np.full(len(model_files), np.inf).tolist()))
    )
    
    # No bound on runtime for default configuration
    if fathoming:
        fathom_times = dict(zip(model_files, np.mean(default_times, axis=2).flatten().tolist()))
    else:
        fathom_times = dict(zip(model_files, np.full(len(model_files), np.inf).tolist()))
    
    times = parallel_solve(
        args=args,
        inputs=build_inputs(actions, model_files, samples),
        num_actions=len(actions),
        num_models=len(model_files),
        num_samples=len(samples),
        fathom_times=fathom_times
    )
    
    return _compute_improv(default_times, times)


def _get_best_action(
    args: Namespace,
    actions: list[np.array],
    model_files: list[str]
) -> tuple:
    improv = evaluate(args, actions, model_files, fathoming=True)
    improv_by_action_instance = np.mean(improv, axis=2)
    improv_by_action = np.median(improv_by_action_instance, axis=1)
    argmax = np.argmax(improv_by_action)
    best_action = actions[argmax]
    best_improv = improv_by_action[argmax]
    return best_action, best_improv


def chunked_evaluate(
    args: Namespace,
    actions: list[np.array],
    model_files: list[str],
    record_fn: callable
) -> tuple:

    chunked_actions = chunk(actions, args.chunk_size)
    iteration = 0
    best_improv = -np.inf
    best_action = None
    print("===========================================================================================================")
    print(f"Finding best [{args.solver}] cutting plane configuration for [{args.eval_type}] instances in [{args.instance_name}]...")
    print(f"Generated {len(actions)} actions to evaluate...")
    print("===========================================================================================================")

    for action_batch in chunked_actions:
        current_action, current_improv = _get_best_action(args, action_batch, model_files)
        iteration += len(action_batch)

        if current_improv > best_improv:
            best_action = current_action
            best_improv = current_improv
        record_fn(args, best_action, best_improv, iteration)
        
        print(f"\niteration {iteration} | best action: {best_action}, best improvement: {np.round(best_improv * 100, 2)}%")
        print(f"last action: {current_action}, last improvement: {np.round(current_improv * 100, 2)}%\n")
