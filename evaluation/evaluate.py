import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser, Namespace
from solve import evaluate
from util import config_to_action, get_model_files
from config_generation.util import RESULTS_DIR, CONFIGS_DIR


def _actions_fn(args: Namespace) -> list[np.array]:
    config_folder = os.path.join(CONFIGS_DIR, args.instance_name, args.solver, args.config_name)
    config_files = np.sort(
        np.array([
            os.path.join(config_folder, file)
            for file in os.listdir(config_folder)
            if file.endswith('.yaml')
        ])
    )
    actions = np.array([config_to_action(file, args.solver) for file in config_files])
    return actions, config_files
        
def _post_process(
    args: Namespace,
    outputs: np.array,
    model_files: list, 
    config_files: list
) -> pd.DataFrame:
    '''
    Outputs a DataFrame with the following format:

    [config_file] | [model_file] | [sample] | [value]
    config1       | model1       | 0        | 0.17
    config1       | model1       | 1        | 0.24
    config1       | model1       | 2        | -0.02
    ...           | ...          | ...      | ...
    '''

    config_indices, model_indices, sample_indices = np.indices(outputs.shape)
    
    df = pd.DataFrame(
        data=np.column_stack(
            (
                config_indices.flatten(),
                model_indices.flatten(),
                sample_indices.flatten(),
                outputs.flatten()
            )
        ),
        columns=["config_file", "model_file", "sample", "value"]
    )

    replace_values = [config_files, model_files, np.arange(args.num_solves)]
    for i, col in enumerate(["config_file", "model_file", "sample"]):
        df[col] = df[col].replace(dict(enumerate(replace_values[i])))
    
    return df

def _save(args: dict, results_df: pd.DataFrame) -> None:
    # Write results to file
    output_folder = os.path.join(RESULTS_DIR, args.instance_name, args.solver, args.config_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    results_df.to_csv(output_folder + f"/{args.eval_type}_raw_scores.csv", index=False)
    

if __name__ == "__main__":
    parser = ArgumentParser()
    # Data settings.
    parser.add_argument('--instance_name', type=str, help="the MILP class")
    parser.add_argument('--config_name', type=str, help="name of configuration")

    # Either solve every instance with every config (agnostic), or each instance with one config (specific).
    parser.add_argument('--eval_type', type=str, help="one of [val] or [eval]", default="eval")

    # Solver settings.
    parser.add_argument('--solver', type=str, help="which MILP solver to use", default='gurobi')
    parser.add_argument('--gap_limit', type=float, default=0.0, help="gap limit for solving instances")
    parser.add_argument('--time_limit', type=float, default=100.0, help="time limit for solving instances")
    parser.add_argument('--num_solves', type=int, default=10, help="number of times to regenerate configuration and solve")
    args = parser.parse_args()
    
    print("===========================================================================================================")
    print(f"Running [{args.config_name}] configs on [{args.instance_name}] instances ({args.eval_type})...")
    print("===========================================================================================================")
  
    actions, configs = _actions_fn(args)
    print(len(actions))
    model_files = get_model_files(args)
    improv = evaluate(args, actions, model_files)
    results_df = _post_process(args, improv, model_files, configs)
    _save(args, results_df)