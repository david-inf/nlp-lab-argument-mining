"""Main program for finetuning a BERT family model"""

import os
import sys
from types import SimpleNamespace

import torch
from torch.optim import AdamW
from torch.backends import cudnn
from accelerate import Accelerator
from transformers import set_seed, get_cosine_schedule_with_warmup, PreTrainedModel

# Ensure the parent directory is in the path for module imports
sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))  # Add parent directory to path

from src.ftdata import get_loaders
from src.models.bert import get_bert
from src.utils.misc_utils import LOG, visualize, update_yaml
from src.train import train_loop


def get_model(opts):
    """Get model to finetune"""
    if opts.model in ("distilbert", "scibert", "sbert"):
        tokenizer, model = get_bert(opts)
    else:
        raise ValueError(f"Unknown model {opts.model}")
    return tokenizer, model


def get_optimization(opts, model: PreTrainedModel, train_loader):
    """Optimizer and LRScheduler settings"""
    configs = SimpleNamespace(**opts.ft_setting)  # dict
    head_params = [p for name, p in model.named_parameters()
                   if "classifier" in name]
    backbone_params = [
        p for name, p in model.named_parameters() if "classifier" not in name]
    params = [
        {"params": head_params, "lr": configs.lr_head},
        {"params": backbone_params},
    ]
    optimizer = AdamW(
        params,
        lr=configs.lr_backbone,
        weight_decay=configs.weight_decay
    )

    total_steps = opts.num_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        int(configs.warmup*total_steps),
        total_steps
    )

    return optimizer, scheduler


def main(opts):
    """BERT model finetuning"""
    set_seed(opts.seed)  # from transformers

    # Checkpointing
    os.makedirs(opts.checkpoint_dir, exist_ok=True)
    output_path = os.path.join(opts.checkpoint_dir, opts.experiment_name)
    update_yaml(opts, "checkpoint", output_path)

    # Accelerator
    accelerator = Accelerator(mixed_precision="fp16", project_dir=output_path,
                              gradient_accumulation_steps=opts.accum_steps)
    LOG.info("Accelerator device=%s", accelerator.device)

    # Get BERT and its tokenizer (cpu)
    tokenizer, model = get_model(opts)
    # Get loaders
    with accelerator.main_process_first():
        train_loader, val_loader = get_loaders(opts, tokenizer)

    # Optimizer
    optimizer, scheduler = get_optimization(opts, model, train_loader)

    # Prepare training -> send to device
    cudnn.benchmark = True
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader)
    accelerator.register_for_checkpointing(scheduler)

    # Training
    LOG.info("Running experiment_name=%s", opts.experiment_name)
    train_loop(opts, model, optimizer, scheduler,
               accelerator, train_loader, val_loader)


def view_model(opts):
    """BERT inspection"""
    # Get BERT and tokenizer
    tokenizer, model = get_model(opts)

    # Random data for input_ids and attention_mask
    input_ids = torch.randint(
        0, tokenizer.vocab_size, (opts.batch_size, 128))
    attention_mask = torch.ones(
        (opts.batch_size, 128), dtype=torch.int64)
    input_data = {"input_ids": input_ids, "attention_mask": attention_mask}

    # Visualize model
    visualize(model, opts.model, input_data)


if __name__ == "__main__":
    from cmd_args import parse_args
    args = parse_args()
    try:
        if args.visualize:
            view_model(args)
        else:
            main(args)
    except Exception:
        import ipdb
        import traceback
        import sys
        traceback.print_exc()
        ipdb.post_mortem(sys.exc_info()[2])
