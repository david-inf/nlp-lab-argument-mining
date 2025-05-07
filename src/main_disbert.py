
# from comet_ml import start
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from accelerate import Accelerator

from models.distilbert import get_distilbert
from ftdata import get_loaders
from utils.misc_utils import set_seeds, LOG, visualize
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
    set_seeds(opts.seed)
    # Get DistilBERT and its tokenizer (cpu)
    tokenizer, model = get_distilbert(opts)
    # Get loaders
    train_loader, val_loader = get_loaders(opts, tokenizer)

    # Prepare training
    cudnn.benchmark = True
    optimizer = optim.AdamW(model.parameters(), lr=opts.learning_rate)
    # TODO: scheduler

    accelerator = Accelerator(mixed_precision="fp16")
    opts.device = accelerator.device
    LOG.info("Accelerator device=%s", opts.device)

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader)

    # Training
    LOG.info("Running experiment_name=%s", opts.experiment_name)
    train_loop(opts, model, optimizer, accelerator, train_loader, val_loader)


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
        import ipdb
        ipdb.post_mortem()
