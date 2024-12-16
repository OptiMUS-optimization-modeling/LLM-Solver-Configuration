import argparse
import os
import numpy as np
import yaml
from pydantic import BaseModel
from util import CONFIGS_DIR
import pandas as pd
from sklearn_extra.cluster import KMedoids 


def aggregate_configs(configs: list[dict]) -> dict:
    ensemble_config = {}

    for config in configs:
        for param in config['parameters']:
            name = param['name']
            value = param['value']
            if name not in ensemble_config:
                ensemble_config[name] = []
            ensemble_config[name].append(value)
    return ensemble_config

def mean_ensemble(configs: list[dict], *args) -> list[dict]:
    '''
    Looks at average usage of each parameter and returns the (rounded) mean value
    '''
    ensemble_config = aggregate_configs(configs)
    num_configs = len(configs)

    empty_config = True
    for key in ensemble_config:
        mean_score = round(np.sum(ensemble_config[key])/num_configs)
        ensemble_config[key] = mean_score
        if mean_score > 0:
            empty_config = False
        
    if empty_config:
        raise ValueError('Ensemble configuration is empty. Please check the input configurations.')
    final_config = {}
    final_config['parameters'] = [{'name': k, 'value': v} for k, v in ensemble_config.items()]
    return [final_config]

def mean_smallest(configs: list[dict], *args) -> dict:
    ensemble_config = {}

    for config in configs:
        if not ensemble_config:
            ensemble_config = config
        else:
            if sum([param['value'] for param in config['parameters']]) < \
                sum([param['value'] for param in ensemble_config['parameters']]):
                ensemble_config = config
    return [ensemble_config]

def mode(configs: list[dict], *args) -> dict:

    config_counts = []
    for id, config in enumerate(configs):
        for param in config['parameters']:
            config_counts.append({
                'config': id,
                'sep': param['name'],
                'value': param['value']
            })

    ct_matrix = pd.DataFrame.from_records(config_counts).pivot_table(index='config', columns='sep', values='value').fillna(0)
    param_dict = ct_matrix.value_counts().reset_index().head(1).drop('count',axis=1).astype(int).to_dict(orient='records')[0]
    ensemble_config = {}
    ensemble_config['parameters'] = [{'name': k, 'value': v} for k, v in param_dict.items()]
    return [ensemble_config]

def kmediods(configs: list[dict], k: int = 5) -> list[dict]:
    config_counts = []
    for id, config in enumerate(configs):
        for param in config['parameters']:
            config_counts.append({
                'config': id,
                'sep': param['name'],
                'value': param['value']
            })

    ct_matrix = pd.DataFrame.from_records(config_counts).pivot_table(index='config', columns='sep', values='value').fillna(0)
    kmed = KMedoids(n_clusters=k, metric='hamming', init='k-medoids++')
    kmed.fit(ct_matrix)
    config_list = []
        
    for i in range(k):
        biggest_cl = pd.Series.value_counts(kmed.labels_).reset_index()['index'][i]
        param_list = []
        for col, val in zip(ct_matrix.columns, kmed.cluster_centers_[biggest_cl]):
            if val > 0:
                param_list.append({
                    'name': col,
                    'value': int(val)
                })
        param_list
        ensemble_config = {'parameters': param_list}
        config_list.append(ensemble_config)

    return config_list

def mean_with_default(configs: list[dict], *args) -> dict:
    ensemble_config = {}
    num_configs = len(configs)
    uncertain_vals = {}
    for config in configs:
        for param in config['parameters']:
            name = param['name']
            value = param['value']
            if value == -1:
                if name not in uncertain_vals:
                    uncertain_vals[name] = 0
                uncertain_vals[name] += 1
            else:
                if name not in ensemble_config:
                    ensemble_config[name] = []
                ensemble_config[name].append(value)

    empty_config = True
    for key in ensemble_config:
        mean_score = np.sum(ensemble_config[key])/num_configs
        if mean_score >= 1.5:
            mean_score = 2
        elif mean_score >= 0.7:
            mean_score = 1
        elif mean_score > 0.3:
            mean_score = -1
        else:
            mean_score = 0
        ensemble_config[key] = mean_score
        if mean_score != 0:
            empty_config = False

    for key in uncertain_vals:
        if key not in ensemble_config:
            ensemble_config[key] = -1
        
    if empty_config:
        raise ValueError('Ensemble configuration is empty. Please check the input configurations.')
    final_config = {}
    final_config['parameters'] = [{'name': k, 'value': v} for k, v in ensemble_config.items()]
    return [final_config]

def union(configs: list[dict], *args) -> dict:
    ensemble_config = aggregate_configs(configs)

    for key in ensemble_config:
        mean_score = max(ensemble_config[key])
        ensemble_config[key] = mean_score
    final_config = {}
    final_config['parameters'] = [{'name': k, 'value': v} for k, v in ensemble_config.items()]
    return [final_config]


def read_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader) 


def ensemble(args: dict) -> list[dict]:
    '''
    Helper function to take a folder of cutting plane configurations and aggregate them together into a single configuration
    '''
    config_folder = os.path.join(CONFIGS_DIR, args.instance_name, args.solver, args.config_name)
    config_paths = [os.path.join(config_folder, file) for file in os.listdir(config_folder) if file.endswith('.yaml')]
    configs = [read_config(path) for path in config_paths]

    ensemble_fns = {
        'mean': mean_ensemble,
        'smallest': mean_smallest,
        'mean_with_default': mean_with_default,
        'mode': mode,
        'kmedoids': kmediods
    }
    try:
        return ensemble_fns[args.ensemble_method](configs, args.k)
    except KeyError:
        raise ValueError('Invalid ensemble method. Select one of: mean, smallest, mean_with_default, mode, kmedoids')
    
    
if __name__ == "__main__":
    #parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, default="generated_configs/", help="directory of the configs")
    parser.add_argument('--instance_name', type=str, help="name of instances", default='all')
    parser.add_argument('--solver', type=str, help="which MILP solver to use", default='gurobi')
    parser.add_argument('--ensemble_method', type=str, help="method to use for ensemble configuration", default='mean')
    parser.add_argument('--k', type=int, help="number of clusters for kmedoids", default=5) 

    args = parser.parse_args()

    write_path = os.path.join(
        CONFIGS_DIR,
        args.instance_name,
        args.solver,
        f"{args.ensemble_method}_{args.config_name}"
    )
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    ensemble_configs = ensemble(args)
    for i, ensemble_config in enumerate(ensemble_configs):
        with open(write_path + f"/{args.config_name}_{i}.yaml", 'w') as f:
            yaml.dump(ensemble_config, f, default_flow_style=False)