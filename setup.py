from setuptools import setup, find_packages

setup(
    name='llm_for_solver_configuration',
    version='0.1',
    packages=find_packages(include=["config_generation", "evaluation"]),
)