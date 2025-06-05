"""Quick program for loading a model and evaluating on test data"""

import sys
import os

import torch
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_from_disk
from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import confusion_matrix, classification_report

# Ensure the parent directory is in the path for module imports
sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))  # Add parent directory to path

from src.utils.train_utils import AverageMeter, accuracy, my_f1_score


def N(x):
    return x.detach().cpu().numpy()


def load_model(opts):
    """Load model from hub"""
    # checkpoint = "david-inf/bert-sci-am"
    if opts.dataset == "abstrct":
        checkpoint = "/data01/dl24davnar/projects/nlp-lab-argument-mining/src/ckpts/sbert_full_abstrct"
    elif opts.dataset == "sciarg":
        checkpoint = "/data01/dl24davnar/projects/nlp-lab-argument-mining/src/ckpts/sbert_full_sciarg"
    elif opts.dataset == "mixed":
        checkpoint = "/data01/dl24davnar/projects/nlp-lab-argument-mining/src/ckpts/sbert_full_mixed"
    elif opts.dataset == "ibm":
        checkpoint = "/data01/dl24davnar/projects/nlp-lab-argument-mining/src/ckpts/sbert_full_ibm"
    else:
        raise ValueError(f"Unknown dataset {opts.dataset}")
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=3)
    # tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    return model, tokenizer


def get_loader(opts, tokenizer):
    if opts.dataset == "abstrct":
        dataset = load_from_disk("data/abstrct")
    elif opts.dataset == "sciarg":
        dataset = load_from_disk("data/sciarg")
    elif opts.dataset == "mixed":
        dataset = load_from_disk("data/mixed")
    elif opts.dataset == "ibm":
        dataset = load_from_disk("data/ibm_dataset")
    else:
        raise ValueError(f"Unknown dataset {opts.dataset}")

    def preprocess(sample):
        return tokenizer(
            # tokenize the text without padding, whatever length
            sample["text"],
            # truncate to specified length if necessary, iff str exceeds
            max_length=128,
            truncation=True,
            return_attention_mask=True,
            # returns lists as the default collator wants
            return_tensors=None,  # return list
        )

    testset = dataset[opts.split_name]
    print(testset)
    testset = testset.map(
        preprocess, batched=True, num_proc=2,
        remove_columns=["text"], desc="Tokenizing")

    collate_fn = DataCollatorWithPadding(
        tokenizer=tokenizer, padding="longest")
    loader = DataLoader(
        testset, batch_size=32, num_workers=2, pin_memory=True, collate_fn=collate_fn)
    return loader


def test(opts, model, loader):
    """Evaluate model on some testset"""
    preds, trues = [], []
    accs, f1s = AverageMeter(), AverageMeter()
    model.eval()
    with torch.no_grad():
        for batch in loader:
            # Get data
            input_ids = batch["input_ids"].to(opts.device)
            attention_mask = batch["attention_mask"].to(opts.device)
            y = batch["labels"].to(opts.device)
            # Forward pass
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            # Metrics
            pred = np.argmax(N(output.logits), axis=1)
            preds.extend(pred.tolist())
            trues.extend(N(y).tolist())

            acc = accuracy(N(output.logits), N(y))
            f1 = my_f1_score(N(output.logits), N(y))
            accs.update(acc, input_ids.size(0))
            f1s.update(f1, input_ids.size(0))

    unique, counts = np.unique(trues, return_counts=True)
    print(unique, counts)
    unique, counts = np.unique(preds, return_counts=True)
    print(unique, counts)
    print(confusion_matrix(trues, preds))
    print(classification_report(trues, preds, digits=3))

    return accs.avg, f1s.avg


def main(opts):
    model, tokenizer = load_model(opts)
    model.to(opts.device)

    val_loader = get_loader(opts, tokenizer)

    val_acc, val_f1 = test(opts, model, val_loader)
    print("val_acc=%.3f" % val_acc)
    print("val_f1=%.3f" % val_f1)


if __name__ == "__main__":
    from types import SimpleNamespace
    configs = {"device": "cuda:1", "dataset": "ibm",
               "split_name": "validation"}
    opts = SimpleNamespace(**configs)
    try:
        main(opts)
    except Exception:
        import ipdb, traceback, sys
        traceback.print_exc()
        ipdb.post_mortem(sys.exc_info()[2])
