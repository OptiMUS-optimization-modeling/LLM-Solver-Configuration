import argparse
import openai
import os
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel
from util import CONFIGS_DIR, DATA_DIR

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG")
if not openai.api_key:
    print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    exit(1)
client = openai.Client()

#Class Template for structured output 
class Parameter(BaseModel):
    name: str
    value: int
    explanation: str

class Configuration(BaseModel):
    parameters: list[Parameter]

def extract_cutting_planes(descriptions: dict, solver: str) -> list[str]:
    solver = str.upper(solver)
    return [
        '- '+sep['solvers'][solver]+' : '+sep['description']
        for sep in descriptions['separators'] 
        if solver in sep['solvers']
    ]

def extract_value_instructions(solver: str) -> str:
    match solver:
        case "gurobi":
            return "A setting of 2 indicates that the separator should be used aggressively, \
                1 indicates use normally, and 0 disables the separator."
        case "scip":
            return "A setting of 1 indicates that the separator should be used, and 0 disables the separator."
        
def extract_default_instructions(allow_default: bool) -> str:
    if allow_default:
        return "You only need to specify cutting planes you are confident you want to turn on or off, \
            all other cutting planes will be set to their default setting."
    else:
        return "You only need to specify cutting planes to turn on, every other separator will be turned off."

def generate_system_prompt(args):
    with open(f'config_generation/cutting_plane_descriptions.yaml', 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    cutting_planes = extract_cutting_planes(data, args.solver)
    value_instructions = extract_value_instructions(args.solver)
    default_instructions = extract_default_instructions(args.allow_default)
   
    system_prompt = f"You are an optimization expert tasked with helping a user select the right cutting planes \
        for solving their optimization problem using {args.solver}. You need to specify which of the following \
        cutting plane separators should turn on: {cutting_planes} {value_instructions} {default_instructions} \
        Recall that running a separator may increase the runtime because it takes to generate the cuts. Use \
        separators you're confident will help the solver find a solution faster."

    return system_prompt

def generate_user_prompt(args):
    path_to_description = os.path.join(DATA_DIR, args.instance_name, "description.txt")
    with open(path_to_description, 'r') as file:
        problem_description = file.read()

    user_prompt = f"Here is the description of the optimization problem: {problem_description}"
    return user_prompt

def generate_config(args):
    system_prompt = generate_system_prompt(args)
    user_prompt = generate_user_prompt(args)

    #Generate config using gpt-4
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=Configuration,
    )
    config = completion.choices[0].message.parsed
    return config


if __name__ == "__main__":
    #parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_name', type=str, help="name of instance family")
    parser.add_argument('--config_name', type=str, help="name of generated configurations", default='baseline')
    parser.add_argument('--allow_default', type=str, default="False", help="allow LLM to pick default settings")
    parser.add_argument('--solver', type=str, help="which MILP solver to use", default='gurobi')
    parser.add_argument('--num_configs', type=int, default=20, help="number configurations to generate")
    

    args = parser.parse_args()
    write_path = os.path.join(CONFIGS_DIR, args.instance_name, args.solver, args.config_name)
    if not os.path.exists(write_path):
        os.makedirs(write_path)
        
    for i in range(args.num_configs):
        print(i)
        print(f"Generating config {i}/{args.num_configs} for {args.instance_name}")
        config = generate_config(args)
        with open(os.path.join(write_path, f'{args.config_name}-{i}.yaml'), 'w') as f:
            yaml.dump(config.dict(), f, default_flow_style=False)
