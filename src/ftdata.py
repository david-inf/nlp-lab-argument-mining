
import os
import sys
import random
import torch
import numpy as np

from torch.utils.data import DataLoader, WeightedRandomSampler
from datasets import load_from_disk, Dataset
from transformers import set_seed, DataCollatorWithPadding, PreTrainedTokenizer

# Ensure the parent directory is in the path for module imports
sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))  # Add parent directory to path

from src.utils import LOG


class MakeDataLoaders:
    """Load data"""

    def __init__(self, opts, tokenizer: PreTrainedTokenizer, trainset: Dataset, valset: Dataset, sampler=None):
        set_seed(opts.seed)
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

        if sampler is None:
            shuffle = True
        else:
            shuffle = False

        self.train_loader = DataLoader(
            trainset, shuffle=shuffle, batch_size=opts.batch_size,
            num_workers=opts.num_workers, pin_memory=True, generator=generator,
            worker_init_fn=seed_worker, collate_fn=collate_fn, sampler=sampler
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
        # dataset = load_dataset("david-inf/am-nlp-abstrct")
        dataset = load_from_disk("data/abstrct")
    elif opts.dataset == "sciarg":
        # dataset = load_dataset("david-inf/am-nlp-sciarg")
        dataset = load_from_disk("data/sciarg")
    elif opts.dataset == "mixed":
        dataset = load_from_disk("data/mixed")
    elif opts.dataset == "ibm":
        dataset = load_from_disk("data/ibm_dataset")
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

    # 3) Class distribution
    unique, counts = np.unique(trainset["label"], return_counts=True)
    size = len(trainset["label"])
    print("Trainset size:", counts, "Total:", counts.sum())
    print("Trainset distrib:", unique, counts / size)

    unique, counts = np.unique(valset["label"], return_counts=True)
    size = len(valset["label"])
    print("Valset size:", counts, "Total:", counts.sum())
    print("Valset distrib:", unique, counts / size)

    # 4) Sampler for class imbalance
    if opts.dataset == "ibm":
        # subsampling the majority class
        class_0 = counts[0]  # data
        class_1 = counts[1]  # evidence
        class_2 = counts[2]  # claim
        mult = 1.5
        weights = np.array([1., mult*class_0/class_1, mult*class_0/class_2])
        # weights = weights / np.sum(weights)  # normalize
        # sampler = WeightedRandomSampler(
        #     weights=weights,
        #     num_samples=int(0.25*len(trainset["label"])),
        #     replacement=True
        # )
        sampler = None
        opts.class_weights = weights.tolist()  # for loss function
        print(opts.class_weights)
    else:
        sampler = None

    # 4) Loaders
    loaders = MakeDataLoaders(opts, tokenizer, trainset, valset, sampler)
    train_loader = loaders.train_loader
    val_loader = loaders.val_loader

    return train_loader, val_loader


def main(opts):
    """Inspect finetuning datasets"""
    # Get tokenizer
    from models import get_bert
    tokenizer: PreTrainedTokenizer = get_bert(opts)[0]
    # Get loaders
    train_loader, val_loader = get_loaders(opts, tokenizer)

    LOG.info("num_batches_train=%s", len(train_loader))
    LOG.info("num_batches_val=%s", len(val_loader))
    for batch_idx, batch in enumerate(train_loader):
        # Get data
        input_ids = batch["input_ids"]
        attn_mask = batch["attention_mask"]
        labels = batch["labels"]
        class_distrib = torch.bincount(labels)
        # Inspect train data
        LOG.info("input_ids=%s\nattention_mask=%s\nlabels=%s",
                 input_ids.shape, attn_mask.shape, labels.shape)
        LOG.info(f"distrib={class_distrib/labels.size(0)}")

        # Inspect first sample
        sample_id = 0
        sample_tokens = tokenizer.convert_ids_to_tokens(input_ids[sample_id])
        sample_tokens_no_pad = [token for token in sample_tokens if token != "[PAD]"]
        LOG.info("%s --> %s", labels[sample_id], sample_tokens_no_pad)
        print()

        if batch_idx == 4:
            break


if __name__ == "__main__":
    from cmd_args import parse_args
    configs = parse_args()
    set_seed(configs.seed)

    try:
        main(configs)
    except Exception:
        import ipdb, traceback, sys
        traceback.print_exc()
        ipdb.post_mortem(sys.exc_info()[2])
