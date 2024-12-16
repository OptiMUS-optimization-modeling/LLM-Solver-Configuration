import configparser

def _get_dir(name: str) -> str:
    config = configparser.ConfigParser()
    config.read("params.ini")
    return config.get("FilePaths", name)


CONFIGS_DIR = _get_dir("configs")
DATA_DIR = _get_dir("data")
RESULTS_DIR = _get_dir("results")
SEARCH_HISTORY_DIR = _get_dir("search_history")
