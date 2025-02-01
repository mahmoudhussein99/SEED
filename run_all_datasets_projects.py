from seed import *

def get_config_for_dataset(config,dataset):
    dataset_split = dataset.split("_")
    if dataset_split[0] =="WDC":
        config | {
            "project_name": dataset,
            "name": "entity_resolution",
            "task_desc": "Given two products, determine whether they are identical product.",
            "inputs": [
                {
                    "name": "entity1",
                    "type": "dict",
                    "desc": "It contains three attributes: `title`, `description`, `brand`. all of the three attributes are strings"
                },
                {
                    "name": "entity2",
                    "type": "dict",
                    "desc": "Same as entity1."
                }
            ],
            "outputs": [
                {
                    "name": "is_same",
                    "type": "bool",
                    "desc": "0 if the two product are not identical, 1 of the two products are identical.",
                    "default": 0
                }
            ],
            "evaluation_metric": "f1",
        }
        config = config | {
                "evaluation_path": "./data/Amazon-Google_demo.jsonl",
                "examples_path": "./data/Amazon-Google_valid.jsonl",
                "codev_examples_path": "data/Amazon-Google_demo.jsonl",
                "codeg_examples_path": "data/Amazon-Google_demo.jsonl",
                "labelled_path": "./data/Amazon-Google_train.jsonl",
                }
    return config
def configurate(root_folder, dataset):
    config = LoadJson(pjoin(root_folder,"config.json"))
    config = get_config_for_dataset(config,dataset)
    SaveJson(config, pjoin(root_folder,"config.json"))


def main():
    datasets=["DBLP_Scholar",
            "WDC_Shoes",
            "WDC_Computers",
            "WDC_Watches",
            "WDC_Cameras",
            "Music_Brainz20k"]

    for dataset in datasets:
        CreateProject(name=dataset, workspace="./projects/")
        root_folder =f"projects/{dataset}"
        configurate(root_folder,dataset)
        CompileProject(root_folder)
        HyperparameterTuning(root_folder)

if __name__ =="__main__":
    main()