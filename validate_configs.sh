#!/bin/bash
source venv/bin/activate

# Extract problem families specified in [params.ini]
config_parse=$(grep -A1000 '^\[Instances\]' "params.ini" | grep -m1 -A1000 '^families' | tail -n +2 | grep -E '^\s+\S')

problem_families=()
while IFS= read -r line; do
    trimmed_line=$(echo "$line" | sed 's/^[[:space:]]*//')
    problem_families+=("$trimmed_line")
done <<< "$config_parse"

gap_limits=(0 0 0 0.1 0 0 0 0 0)
config_name="kmedoids_llm"
solvers=(gurobi scip)
num_solves=10

# Run configurations on validation set
for solver in ${solvers[@]}; do
    for i in ${!problem_families[@]}; do
        nohup python3 evaluation/evaluate.py \
            --instance_name ${problem_families[$i]} \
            --config_name $config_name \
            --eval_type val \
            --solver $solver \
            --num_solves $num_solves \
            --gap_limit ${gap_limits[$i]} \
            >> logs/evaluation.log 2> /dev/null
    done
done

# Extract configuration with best median validation performance
for solver in ${solvers[@]}; do
    for i in ${!problem_families[@]}; do
        nohup python3 config_generation/validate.py \
            --instance_name ${problem_families[$i]} \
            --config_name $config_name \
            --solver $solver \
            >> logs/evaluation.log 2> logs/evaluation.log
    done
done

kill -9 `jobs -ps`