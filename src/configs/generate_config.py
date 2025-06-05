"""Generate YAML configuration files"""

import os
import yaml


def gen_configs(new_params):
    """Generate a configuration file given the params dict"""
    configs = {
        "seed": 42, "batch_size": 32, "num_workers": 4, "max_length": 128,
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
        output_dir = "src/ckpts"
        configs["checkpoint_dir"] = output_dir
    os.makedirs(configs["checkpoint_dir"], exist_ok=True)

    # Dump configuration file
    fname = configs["experiment_name"] + ".yaml"
    output_dir = os.path.join("src/configs")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, fname)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(configs, f)

    print(f"Generated file: {output_path}")
    print(f"Configs: {configs}")
    print(f"Experiment: {configs["experiment_name"]}")


if __name__ == "__main__":
    MODEL = "distilbert"
    DATASET = "sciarg"
    NUM_EPOCHS = 50
    ALPHA = 32
    EARLY_STOPPING = {"patience": 7, "min_delta": 0.001}
    new_configs = [
        # Full finetuning
        {"model": MODEL, "num_epochs": NUM_EPOCHS, "dataset": DATASET,
         "early_stopping": EARLY_STOPPING, "accum_steps": 2,
         "ft_setting": {
             "type": "full", "ftname": "full", "lr_head": 5e-5,
             "lr_backbone": 5e-6, "weight_decay": 0.001, "warmup": 0.05,
         }
         },
        # LoRA
        #  {"model": MODEL, "num_epochs": NUM_EPOCHS, "dataset": DATASET,
        #  "early_stopping": EARLY_STOPPING, "accum_steps": 4,
        #  "ft_setting": {
        #      "type": "lora", "rank": 16, "alpha": ALPHA, "target_modules": ["q"],
        #      "lr_head": 0.0001, "lr_backbone": 5e-5, "weight_decay": 0.001,
        #      "warmup": 0.05, "ftname": "lora_q16"
        #  },
        #  "log_every": 100,
        #  },
    ]

    for params_dict in new_configs:
        gen_configs(params_dict)
        print()
    print("Done!")
