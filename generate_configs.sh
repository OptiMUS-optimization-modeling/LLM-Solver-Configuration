#!/bin/bash
source venv/bin/activate

# Extract problem families specified in [params.ini]
config_parse=$(grep -A1000 '^\[Instances\]' "params.ini" | grep -m1 -A1000 '^families' | tail -n +2 | grep -E '^\s+\S')

problem_families=()
while IFS= read -r line; do
    trimmed_line=$(echo "$line" | sed 's/^[[:space:]]*//')
    problem_families+=("$trimmed_line")
done <<< "$config_parse"


solvers=(gurobi scip)
num_configs=100
config_name=llm
allow_default=False

for solver in ${solvers[@]}; do
    for family in ${problem_families[@]}; do
        nohup python3 config_generation/llm.py \
            --instance_name $family \
            --config_name $config_name \
            --allow_default $allow_default \
            --solver $solver \
            --num_configs $num_configs \
            >> logs/generate_configs.log 2> logs/generate_configs.log
    done
done

kill -9 `jobs -ps`