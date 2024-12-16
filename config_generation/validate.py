import argparse
import os
import pandas as pd
import shutil
from util import CONFIGS_DIR, RESULTS_DIR

def _save_best_config(args: argparse.Namespace) -> None:
    results_path = f"{RESULTS_DIR}{args.instance_name}/{args.solver}/{args.config_name}/val_raw_scores.csv"
    new_config_name = f"val_{args.config_name}"
    save_path = f"{CONFIGS_DIR}{args.instance_name}/{args.solver}/{new_config_name}/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df = pd.read_csv(results_path)
    best_config_file = df \
        .groupby(['model_file', 'config_file']) \
        .agg({'value': 'mean'}) \
        .groupby('config_file').median() \
        .sort_values(by='value', ascending=False) \
        .index[0]
    shutil.copy(best_config_file, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_name', type=str, help="the MILP class")
    parser.add_argument('--config_name', type=str, help="name of configuration")
    parser.add_argument('--solver', type=str, help="which MILP solver to use", default='gurobi')
    args = parser.parse_args()

    _save_best_config(args)
    
    
    