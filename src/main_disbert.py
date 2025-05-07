
import os
# from comet_ml import start
import torch
from torch import optim
from torch.backends import cudnn
from accelerate import Accelerator
from transformers import set_seed, get_cosine_schedule_with_warmup

from ftdata import get_loaders
from models.distilbert import get_distilbert
from utils.misc_utils import LOG, visualize, update_yaml
from train import train_loop


def get_model(opts):
    """Get DistilBERT"""
    if opts.model == "distilbert":
        tokenizer, model = get_distilbert(opts)
    else:
        raise ValueError(f"Unknown model {opts.model}")
    return tokenizer, model


def main(opts):
    """DistilBERT finetuning"""
    set_seed(opts.seed)

    # Checkpointing
    os.makedirs(opts.checkpoint_dir, exist_ok=True)
    output_path = os.path.join(opts.checkpoint_dir, opts.experiment_name)
    update_yaml(opts, "checkpoint", output_path)

    # Accelerator
    accelerator = Accelerator(mixed_precision="fp16", project_dir=output_path)
    # opts.device = accelerator.device
    # LOG.info("Accelerator device=%s", opts.device)

    # Get DistilBERT and its tokenizer (cpu)
    tokenizer, model = get_distilbert(opts)
    # Get loaders
    with accelerator.main_process_first():
        train_loader, val_loader = get_loaders(opts, tokenizer)

    # Prepare training
    cudnn.benchmark = True
    optimizer = optim.AdamW(
        model.parameters(), lr=opts.learning_rate, weight_decay=opts.weight_decay)
    total_steps = opts.num_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(opts.warmup*total_steps), total_steps)

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader)
    accelerator.register_for_checkpointing(scheduler)

    # Training
    LOG.info("Running experiment_name=%s", opts.experiment_name)
    train_loop(opts, model, optimizer, scheduler, accelerator, train_loader, val_loader)


def view_model(opts):
    """DistilBERT inspection"""
    # Get BERT and tokenizer
    tokenizer, model = get_model(opts)
    # Random data for input_ids and attention_mask
    input_ids = torch.randint(
        0, tokenizer.vocab_size, (opts.batch_size, 128)).to(opts.device)
    attention_mask = torch.ones(
        (opts.batch_size, 128), dtype=torch.int64).to(opts.device)
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
        import ipdb, traceback, sys
        traceback.print_exc()
        ipdb.post_mortem(sys.exc_info()[2])
