
from types import SimpleNamespace
import argparse
import yaml


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="src/configs/config.yaml",
                    help="YAML configuration file")
parser.add_argument("--view", action="store_true",  # default False
                    help="Visualize model architecture, no training")


def parse_args():
    """Parse arguments given via CLI"""
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)  # return dict
    opts = SimpleNamespace(**configs)

    opts.visualize = args.view
    opts.config = args.config

    return opts
