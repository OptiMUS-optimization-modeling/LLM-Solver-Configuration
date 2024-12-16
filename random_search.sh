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
solvers=(gurobi scip)
num_configs=500
num_solves=10
chunk_size=50

for i in ${!problem_families[@]}; do
    for solver in ${solvers[@]}; do
        nohup python3 config_generation/random_search.py \
            --instance_name ${problem_families[$i]} \
            --num_configs $num_configs \
            --num_solves $num_solves \
            --chunk_size $chunk_size \
            --gap_limit ${gap_limits[$i]} \
            --solver $solver \
            >> logs/random_search.log 2> /dev/null
    done
done

kill -9 `jobs -ps`
