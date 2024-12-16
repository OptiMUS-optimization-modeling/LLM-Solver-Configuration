import argparse
import configparser
import multiprocessing as mp
import numpy as np
import os
import yaml
from itertools import islice, product
from config_generation.util import DATA_DIR

def chunk(lst: list, n: int) -> list[list]:
    iterator = iter(lst)
    return iter(lambda: list(islice(iterator, n)), [])

def get_cpu_count(solver: str) -> int:
    match solver:
        case "gurobi":
            return mp.cpu_count() // 4
        case "scip":
            return mp.cpu_count() // 4
        case _:
            raise NotImplementedError(f"Solver {solver} is not supported right now...")

def get_cutting_planes(solver: str) -> list[str]:
    config = configparser.ConfigParser()
    config.read("params.ini")
    try:
        return config.get("CuttingPlanes", solver).split()
    except:
        raise NotImplementedError(
            f"Solver {solver} is not supported right now... select from [gurobi] or [scip]"
        )
    
def build_inputs(
    actions: list[np.array],
    model_files: list[str],
    samples: np.array
) -> list[tuple]:
    return list(product(actions, model_files, samples))

def get_model_files(args: argparse.Namespace) -> list[str]:
    match args.eval_type:
        case "eval":
            subfolder = "eval_instances"
        case "val":
            subfolder = "val_instances"
        case _:
            raise NotImplementedError(f"Evaluation type {args.eval_type} is not supported right now...")
        
    model_folder = os.path.join(DATA_DIR, args.instance_name, subfolder)
    return np.sort([
        os.path.join(model_folder, f)
        for f in os.listdir(model_folder)
        if f.endswith('.mps') or f.endswith('.lp')
    ])


def get_all_instance_families() -> list[str]:
    config = configparser.ConfigParser()
    config.read("params.ini")
    return config.get("Instances", "families").split()


def _gurobi_config_to_action(config: dict) -> np.array:
 
    # Initialize all cutting planes to off
    all_off = dict(zip(get_cutting_planes("gurobi"), [0] * len(get_cutting_planes("gurobi"))))

    # Set the cutting planes that are on in the config file
    for param in config['parameters']:
        if param['name'] in all_off:
            if param['value'] > 0:
                all_off[param['name']] = param['value']
            elif param['value'] < 0:
                all_off[param['name']] = -1

    return np.array(list(all_off.values()))

def _scip_config_to_action(config: dict) -> np.array:

    # Initialize all cutting planes to off
    all_off = dict(zip(get_cutting_planes("scip"), [0] * len(get_cutting_planes("scip"))))

    # Set the cutting planes that are on in the config file
    for param in config['parameters']:
        if param['name'] in all_off and param['value'] >= 0.5:
            all_off[param['name']] = param['value']

    return np.array(list(all_off.values()))

def config_to_action(config_file: str, solver: str) -> np.array:
    with open(config_file, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    match solver:
        case "gurobi":
            return _gurobi_config_to_action(config)
        case "scip":
            return _scip_config_to_action(config)
        case _:
            raise NotImplementedError(f"Solver {solver} is not supported right now...")
        
def action_to_config(action: np.array, solver: str) -> dict:
    match solver:
        case "gurobi":
            cutting_planes = get_cutting_planes("gurobi")
        case "scip":
            cutting_planes = get_cutting_planes("scip")
    
    return {
        "solver": solver,
        "parameters": [
            {"name": cutting_planes[i], "value": action[i]}
            for i in range(len(action))
            if action[i] > 0
        ]
    }


