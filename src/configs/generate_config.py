"""Generate YAML configuration files"""

import os
import yaml


def gen_configs(new_params):
    """Generate a configuration file given the params dict"""
    configs = {
        "seed": 42, "batch_size": 32, "num_workers": 4,
        "checkpoint": None, "checkpoint_every": None, "log_every": 20,
    }
    configs.update(new_params)

    # Experiment naming
    if configs.get("experiment_name") is None:
        _ft = configs["ft_setting"]["ftname"]
        exp_name = f"{configs["model"]}_{_ft}_{configs["dataset"]}"
        configs["experiment_name"] = exp_name

    # Checkpoint directory
    if configs.get("checkpoint_dir") is None:
        output_dir = os.path.join("src/ckpts", configs["model"])
        configs["checkpoint_dir"] = output_dir
    os.makedirs(configs["checkpoint_dir"], exist_ok=True)

    # Dump configuration file
    fname = configs["experiment_name"] + ".yaml"
    output_dir = os.path.join("src/configs", configs["model"])
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, fname)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(configs, f)

    print(f"Generated file: {output_path}")
    print(f"Configs: {configs}")
    print(f"Experiment: {configs["experiment_name"]}")


if __name__ == "__main__":
    MODEL = "distilbert"
    DATASET = "abstrct"
    NUM_EPOCHS = 50

    new_configs = [
        # Full finetuning
        {"model": MODEL, "dataset": DATASET, "max_length": 128,
         "num_epochs": NUM_EPOCHS, "learning_rate": 5e-5, "weight_decay": 0.01, "warmup": 0.05,
         "early_stopping": {"patience": 5, "min_delta": 0.002},
         "ft_setting": {
             "type": "full", "ftname": "full", "lr_head": 0.001
         }
         },
        # LoRA finetuning
        # {"model": MODEL, "dataset": DATASET, "max_length": 128,
        #  "num_epochs": NUM_EPOCHS, "learning_rate": 5e-5, "weight_decay": 0.01, "warmup": 0.05,
        #  "early_stopping": {"patience": 5, "min_delta": 0.002},
        #  "ft_setting": {
        #      "type": "lora", "rank": 2, "alpha": 8, "target_modules": ["q_lin"],
        #      "ftname": "lora_q_2"},
        # },
        # {"model": MODEL, "dataset": DATASET, "max_length": 128,
        #  "num_epochs": NUM_EPOCHS, "learning_rate": 5e-5, "weight_decay": 0.01, "warmup": 0.05,
        #  "early_stopping": {"patience": 5, "min_delta": 0.002},
        #  "ft_setting": {
        #      "type": "lora", "ftname": "lora_qv_4", "rank": 2, "alpha": 8,
        #      "target_modules": ["q_lin", "v_lin"],
        #      "lr_head": 0.01,
        #      },
        # },
    ]

    for params_dict in new_configs:
        gen_configs(params_dict)
        print()
    print("Done!")
