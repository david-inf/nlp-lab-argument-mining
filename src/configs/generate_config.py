
import os
import yaml


def gen_configs(new_params):
    """Generate configuration file given the params dict"""
    configs = {
        "seed": 42, "batch_size": 32, "num_workers": 4,
        "num_epochs": 5,
        "checkpoint": None, "checkpoint_every": None, "log_every": 20,
        # "comet_project":, "experiment_key":, 
    }
    configs.update(new_params)

    # Experiment naming
    if configs.get("experiment_name") is None:
        _ft = configs["ft_setting"]["type"]
        exp_name = f"{configs["model"]}_{_ft}"
        configs["experiment_name"] = exp_name

    # Checkpoint directory
    if configs.get("checkpoint_dir") is None:
        output_dir = os.path.join("src/ckps", configs["model"])
        os.makedirs(output_dir, exist_ok=True)
        configs["checkpoint_dir"] = output_dir

    # Dump configuration file
    fname = configs["experiment_name"] + ".yaml"
    output_dir = os.path.join("src/configs", configs["model"])
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, fname)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(configs, f)

    print(f"Generated file: {output_path}")
    print(f"Configs: {configs}")


if __name__ == "__main__":
    new_configs = [
        # DistilBERT - Head
        # {"model": "distilbert", "dataset": "abstrct", "max_length": 128,
        #  "num_epochs": 10, "learning_rate": 5e-5, "weight_decay": 0.01, "warmup": 0.05,
        #  "ft_setting": {"type": "head"}, "early_stopping": {"patience": 4, "min_delta": 0.003},
        # },
        # DistilBERT/SciBERT - Full
        # {"model": "scibert", "dataset": "abstrct", "max_length": 128,
        #  "num_epochs": 10, "learning_rate": 5e-5, "weight_decay": 0.01, "warmup": 0.05,
        #  "ft_setting": {"type": "full"}, "early_stopping": {"patience": 4, "min_delta": 0.003},
        # },
        # DistilBERT LoRA
        # {"model": "distilbert", "dataset": "abstrct", "max_length": 128,
        #  "num_epochs": 10, "learning_rate": 5e-5, "weight_decay": 0.01, "warmup": 0.05,
        #  "early_stopping": {"patience": 4, "min_delta": 0.003},
        #  "ft_setting": {
        #      "type": "lora", "rank": 2, "alpha": 8, "target_modules": ["q_lin"]},
        #  "experiment_name": "distilbert_lora_q_2",
        # },
    ]

    for params_dict in new_configs:
        gen_configs(params_dict)
        print()
    print("Done!")
