
import torch
import random
import numpy as np
from types import SimpleNamespace

from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer

from utils import set_seeds, LOG
from models.distilbert import get_distilbert


class MakeDataLoaders:
    """Load data"""
    def __init__(self, opts, tokenizer: PreTrainedTokenizer, trainset: Dataset, valset: Dataset):
        set_seeds(opts.seed)
        generator = torch.Generator().manual_seed(opts.seed)
        collate_fn = DataCollatorWithPadding(
            tokenizer=tokenizer,
            # dynamic padding, different per each batch
            padding="longest"
        )

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        self.train_loader = DataLoader(
            trainset, shuffle=True, batch_size=opts.batch_size,
            num_workers=opts.num_workers, pin_memory=True, generator=generator,
            worker_init_fn=seed_worker, collate_fn=collate_fn
        )
        self.val_loader = DataLoader(
            valset, batch_size=opts.batch_size, num_workers=opts.num_workers,
            pin_memory=True, generator=generator, worker_init_fn=seed_worker,
            collate_fn=collate_fn
        )


def get_loaders(opts, tokenizer: PreTrainedTokenizer):
    """Load finetuning dataset"""
    # 1) Get dataset splits
    if opts.dataset == "abstrct":
        dataset = load_from_disk("data/finetuning/abstrct")
    else:
        raise ValueError(f"Unknown dataset {opts.dataset}")

    # 2) Preprocess data
    def preprocess(sample):
        return tokenizer(
            # tokenize the text without padding, whatever length
            sample["text"],
            # truncate to specified length if necessary, iff str exceeds
            max_length=opts.max_length,  # 128
            truncation=True,
            return_attention_mask=True,
            # returns lists as the default collator wants
            return_tensors=None,  # return list
        )

    tokenized_dataset = dataset.map(
        preprocess, batched=True, num_proc=2,
        remove_columns=["text"], desc="Tokenizing")
    trainset = tokenized_dataset["train"]
    valset = tokenized_dataset["validation"]

    # 3) Loaders
    loaders = MakeDataLoaders(opts, tokenizer, trainset, valset)
    train_loader = loaders.train_loader
    val_loader = loaders.val_loader

    return train_loader, val_loader


def main(opts):
    # Get tokenizer
    tokenizer, _ = get_distilbert(opts)
    # Get dataloaders
    train_loader, val_loader = get_loaders(opts, tokenizer)

    LOG.info("Train data")
    for batch_idx, batch in enumerate(train_loader):
        # Get data
        input_ids = batch["input_ids"].to(opts.device)
        attn_mask = batch["attention_mask"].to(opts.device)
        labels = batch["labels"].to(opts.device)
        class_distrib = torch.bincount(labels)
        # Inspect train data
        LOG.info("input_ids=%s\nattention_mask=%s\nlabels=%s",
                 input_ids.shape, attn_mask.shape, labels.shape)
        LOG.info(f"distrib={class_distrib/labels.size(0)}")

        if batch_idx == 2:
            break
    print()
    LOG.info("Validation data")
    for batch_idx, batch in enumerate(val_loader):
        # Get data
        input_ids = batch["input_ids"].to(opts.device)
        attn_mask = batch["attention_mask"].to(opts.device)
        labels = batch["labels"].to(opts.device)
        class_distrib = torch.bincount(labels)
        # Inspect validation data
        LOG.info("input_ids=%s\nattention_mask=%s\nlabels=%s",
                 input_ids.shape, attn_mask.shape, labels.shape)
        LOG.info(f"distrib={class_distrib/labels.size(0)}")

        if batch_idx == 2:
            break


if __name__ == "__main__":
    configs = {
        "seed": 42, "batch_size": 32, "num_workers": 2, "device": "cpu",
        "dataset": "abstrct", "max_length": 128,
        "model": "distilbert", "ft_setting": {"type": "head"},
    }
    configs = SimpleNamespace(**configs)
    set_seeds(configs.seed)

    try:
        main(configs)
    except Exception:
        from ipdb import set_trace
        set_trace()
