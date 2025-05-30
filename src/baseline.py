"""Baseline to improve with finetuning"""

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from datasets import load_from_disk
from transformers import set_seed, pipeline

from rich.console import Console


def get_dataset(opts):
    """Dataset splits"""
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
    return dataset["train"], dataset["validation"]


def bert_features(texts, bert_name):
    """Use DistilBERT/SciBERT as feature extractor"""
    if bert_name == "distilbert":
        checkpoint = "distilbert-base-uncased"
    elif bert_name == "scibert":
        checkpoint = "allenai/scibert_scivocab_uncased"
    elif bert_name == "sbert":
        checkpoint = "sentence-transformers/all-mpnet-base-v2"
    else:
        raise ValueError(f"Unknown BERT model {bert_name}")

    feature_extractor = pipeline(
        model=checkpoint, tokenizer=checkpoint, task="feature-extraction",
        device_map="auto", framework="pt", batch_size=32,
        tokenize_kwargs=dict(max_length=128, truncation=True))
    # extract features
    extractions = feature_extractor(texts, return_tensors="pt")

    features = []
    for extract in extractions:  # get a tensor
        # extract CLS token
        features.append(extract[0].numpy()[0])
    return np.vstack((features))


def main(opts):
    """Extract features and train classifier"""
    set_seed(opts.seed)
    # Get train-val split
    trainset, valset = get_dataset(opts)

    train_labels = np.array(trainset["label"])
    val_labels = np.array(valset["label"])
    # Extract features
    if opts.extractor in ("distilbert", "scibert", "sbert"):
        train_features = bert_features(trainset["text"], opts.extractor)
        val_features = bert_features(valset["text"], opts.extractor)
    else:
        raise ValueError(f"Unknown extractor {opts.extractor}")

    console = Console()
    # Train classifier and do inference
    console.print("LogisticRegression classifier")
    clf = LogisticRegression(max_iter=250)

    # Fit and compute metrics
    clf.fit(train_features, train_labels)

    console = Console()
    report = classification_report(train_labels,
             clf.predict(train_features), digits=3)
    console.print("Training report")
    console.print(report)

    report = classification_report(
        val_labels, clf.predict(val_features), digits=3)
    console.print("Validation report")
    console.print(report)


if __name__ == "__main__":
    from types import SimpleNamespace
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="abstrct",
                        choices=["abstrct", "sciarg", "mixed", "ibm"],
                        help="Choose dataset to obtain baseline with")
    parser.add_argument("--extractor", default="distilbert",
                        choices=["distilbert", "scibert", "sbert"],
                        help="Choose the feature extractor")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    configs = dict(seed=42, dataset=args.dataset, batch_size=args.batch_size, device="cuda",
                   extractor=args.extractor)
    args = SimpleNamespace(**configs)

    try:
        main(args)
    except Exception:
        import ipdb
        ipdb.post_mortem()
