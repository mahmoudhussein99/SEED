from seed import *

def get_config_for_dataset(config,dataset,efficient=False):
    dataset_split = dataset.split("_")
    if dataset_split[0] =="WDC":
        config = config | {
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
                "evaluation_path": f"/scratch/mhussein/data/WDC/seed/{dataset_split[1]}/test.jsonl" if not efficient else f"/scratch/mhussein/data/WDC/seed/{dataset_split[1]}/test_downsampled.jsonl",
                "examples_path": f"/scratch/mhussein/data/WDC/seed/{dataset_split[1]}/validation.jsonl",
                "codev_examples_path": f"/scratch/mhussein/data/WDC/seed/{dataset_split[1]}/validation.jsonl",
                "codeg_examples_path": f"/scratch/mhussein/data/WDC/seed/{dataset_split[1]}/validation.jsonl",
                "labelled_path": f"/scratch/mhussein/data/WDC/seed/{dataset_split[1]}/train.jsonl",
                }
        return config
    elif dataset_split[0]=="DBLP":
        config = config | {
            "project_name": dataset,
            "name": "entity_resolution",
            "task_desc": "Given two bibliographic records, determine whether they are identical records.",
            "inputs": [
                {
                    "name": "entity1",
                    "type": "dict",
                    "desc": "It contains two attributes: `title`, `authors`. Both attributes are strings"
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
                    "desc": "0 if the two records are not identical, 1 if the two records are identical.",
                    "default": 0
                }
            ],
            "evaluation_metric": "f1",
        }
        config = config | {
                "evaluation_path": f"/scratch/mhussein/data/DBLP-Scholar/seed/test.jsonl" if not efficient else f"/scratch/mhussein/data/DBLP-Scholar/seed/test_downsampled.jsonl",
                "examples_path": f"/scratch/mhussein/data/DBLP-Scholar/seed/validation.jsonl",
                "codev_examples_path": f"/scratch/mhussein/data/DBLP-Scholar/seed/validation.jsonl",
                "codeg_examples_path": f"/scratch/mhussein/data/DBLP-Scholar/seed/validation.jsonl",
                "labelled_path": f"/scratch/mhussein/data/DBLP-Scholar/seed/train.jsonl",
                }

        return config
    elif dataset_split[0]=="Music":
        config = config | {
            "project_name": dataset,
            "name": "entity_resolution",
            "task_desc": "Given two musical metadata records, determine whether they are identical records.",
            "inputs": [
                {
                    "name": "entity1",
                    "type": "dict",
                    "desc": "It contains four attributes: `title`, `artist`, `album`, `year`. The first three attributes `title`, `artist` and `album` are strings and the `year` attribute is an integer"
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
                    "desc": "0 if the two records are not identical, 1 if the two records are identical.",
                    "default": 0
                }
            ],
            "evaluation_metric": "f1",
        }
        config = config | {
                "evaluation_path": f"/scratch/mhussein/data/MusicBrainz/seed/test.jsonl"if not efficient else f"/scratch/mhussein/data/MusicBrainz/seed/test_downsampled.jsonl",
                "examples_path": f"/scratch/mhussein/data/MusicBrainz/seed/validation.jsonl",
                "codev_examples_path": f"/scratch/mhussein/data/MusicBrainz/seed/validation.jsonl",
                "codeg_examples_path": f"/scratch/mhussein/data/MusicBrainz/seed/validation.jsonl",
                "labelled_path": f"/scratch/mhussein/data/MusicBrainz/seed/train.jsonl",
                }

        return config
    else:
        raise Exception(f"The following dataset isn't supported yet {dataset} ")

def configurate(root_folder, dataset,efficient=False):
    config = LoadJson(pjoin(root_folder,"config.json"))
    config = get_config_for_dataset(config,dataset,efficient)
    SaveJson(config, pjoin(root_folder,"config.json"))


def main():
    datasets=[
        # "DBLP_Scholar",
            "WDC_shoes",
            "WDC_computers",
            "WDC_watches",
            "WDC_cameras",
            "Music_Brainz20k"]
    efficient=True
    for dataset in datasets:
        CreateProject(name=dataset, workspace="./projects/")
        root_folder =f"projects/{dataset}"
        configurate(root_folder,dataset,efficient=efficient)
        CompileProject(root_folder, best=True)
        vari_config = LoadJson(pjoin(root_folder, 'config.json'))
        profile = evaluate_config(root_folder, vari_config)
        add_config(root_folder, vari_config, profile,counter=0)
        # HyperparameterTuning(root_folder,efficient=efficient)

if __name__ =="__main__":
    main()