#!/bin/bash
rm logs/ensemble_configs.log
source venv/bin/activate

# Extract problem families specified in [params.ini]
config_parse=$(grep -A1000 '^\[Instances\]' "params.ini" | grep -m1 -A1000 '^families' | tail -n +2 | grep -E '^\s+\S')

problem_families=()
while IFS= read -r line; do
    trimmed_line=$(echo "$line" | sed 's/^[[:space:]]*//')
    problem_families+=("$trimmed_line")
done <<< "$config_parse"

solvers=(gurobi scip)
config_name=llm
ensemble_method=kmedoids
k=5

for solver in ${solvers[@]}; do
    for family in ${problem_families[@]}; do
        nohup python3 config_generation/ensemble.py \
            --instance_name $family \
            --config_name $config_name \
            --solver $solver \
            --ensemble_method $ensemble_method \
            --k $k \
            >> logs/ensemble_configs.log 2> /dev/null
    done
done

kill -9 `jobs -ps`