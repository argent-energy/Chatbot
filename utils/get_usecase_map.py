import yaml
import os

def load_usecase_map(taxonomy_config):
    """Reads a YAML file into a dictionary.

    Args:
      yaml_file_path: The path to the YAML file.

    Returns:
      A dictionary containing the contents of the YAML file.
    """

    # taxonomy_config = f"config/{os.environ['TAXONOMY_CONFIG']}.yaml"
    with open(taxonomy_config, "r") as f:
        usecase_map = yaml.safe_load(f)

    return usecase_map

def get_usecase_categories(map):
    """
    Returns a list of all the categories in the hierarchy.json file.
    """
    categories = []
    for i in map["categories"]:
        if i["name"] != "":
            categories.append(i["name"])
    return categories

def get_usecase_functions(map, category):
    """
    Returns a dictionary of all the usecases in a given category as .
    """
    usecases = {}
    for i in map["categories"]:
        if i["name"] == category:
            for j in i["usecases"]:
                usecases[j["name"]] = j["form"]
    return usecases
