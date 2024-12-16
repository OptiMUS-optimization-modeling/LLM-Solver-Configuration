# LLM for Solver Configuration

## Setup
This repository contains code for LLM-based cutting plane separator configuration. A short setup (3 steps) is required to begin running code.

(1) **Set up the Python environment using virtualenv**. While in the root of the project, run the command `virtualenv venv` to create a fresh environment, then `source venv/bin/activate` to activate it. Once activated, install all required packages using `pip3 install -r requirements.txt`. Finally, install the codebase itself as a package to be able to access config generation and evaluation as subpackages: `pip3 install .`. Optionally, add a `-e` flag to the end of the previous command to install these packages in "editable" mode.

(2) **Add OpenAI API access information**. Create a file called `.env` and add two lines: the first `OPENAI_API_KEY="{key}"` gives your (personal) API access key, while the second `OPENAI_ORG="{org}"` gives the organization that is billed for queries. These will be read into the Python environment automatically.

(3) **Download MILP datasets**. Download the data folder available here [need some sort of public storage] and copy its contents into the empty `data/` directory.

## Generating separator configurations
`generate_config.sh` executes code to generate llm-based configurations (`config_generation/llm.py`), while `random_search.sh` executes code to generate the best configuration found during the random search procedure (`config_generation/random_search.py`).

The inputs to `generate_config.sh` include:
- `config_name`: A name for the collection of configurations.
- `num_configs`: the number of configurations to have the llm generate.
- `allow_default`: Whether to allow the llm to decide to use default settings for any cutting plane family.
```
bash generate_configs.sh
```
Meanwhile, the inputs to `random_search.py` include:
- `num_configs`: The depth of the search, i.e., the number of configurations randomly sampled and tested.
- `num_solves`: the number of times each MILP is solved (runtimes averaged) to find the configuration with the best overall runtime.
- `chunk_size`: The maximum number of configurations to evaluate in parallel. Progress will be saved after each batch of size [chunk_size].
```
bash random_search.sh
```
The resulting configurations can be found in the directories `configs/{instance_family}/{solver}{config_name}/` and `configs/{instance_family}/{solver}/random_search/`, respectively.

## Ensembling configurations
`ensemble_configs.sh` runs code for ensembling collection of configurations (`config_generation/emsemble.py`). That is, generating a single, representative config from a collection of configs. The inputs to the script include:
- `config_name`: The name of the collection of configurations to ensemble (this should exactly match the output from the generation step).
- `ensemble_method`: One of "mean", "smallest", "mean_with_default", "mode", "kmedoids" 
- `k`: The number of clusters (only necessary when "kmedoids" is the given ensemble type)

```
bash ensemble_configs.sh
```
A new folder of ensembled configs will be generated in the same directory under the name `{ensemble_method}_{config_name}/`.

## Choosing the best from a folder of configurations
`validate_configs.sh` runs code for evaluating a collection of configurations on a validation set (`evaluation/evaluate.py`), then selecting the one with the best performance (`config_generation/validate.py`). The inputs to the script include:
- `config_name`: The name of the collection of configurations to evaluate.
- `num_solves`: The number of solves per MILP instance during performance evaluation.

```
bash validate_configs.sh
```
A new folder containing the best configuration will be generated in the same directory under the name `val_{config_name}/`

## Evaluating configurations
`evaluate_configs.sh` runs code for evaluating a collection of configurations on the evaluation set of MILPs (`evaluation/evaluate.py`). The inputs to the script include:
- `config_name`: The name of the collection of configurations to evaluate.
- `num_solves`: The number of solves per MILP instance during performance evaluation.

```
bash evaluate_configs.sh
```
A .csv file will be generated in the directory `results/{instance_family}/{solver}/{config_name}/` that gives the runtime improvement of each configuration in the collection on every instance in the evaluation set, for each of the [num_solves] solves. We use the information available in these files to generate all figures in the paper.

