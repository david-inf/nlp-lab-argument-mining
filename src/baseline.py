"""Baseline to improve with finetuning"""

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, balanced_accuracy_score

from datasets import load_dataset
from transformers import set_seed, pipeline

from utils.misc_utils import LOG


def get_dataset(opts):
    """Dataset splits"""
    if opts.dataset == "abstrct":
        dataset = load_dataset("david-inf/am-nlp-abstrct")
    elif opts.dataset == "sciarg":
        dataset = load_dataset("david-inf/am-nlp-sciarg")
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
        checkpoint = "sentence-transformers/all-MiniLM-L6-v2"
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

    # Train classifier and do inference
    LOG.info("LogisticRegression classifier")
    clf = LogisticRegression()

    # Fit and compute metrics
    clf.fit(train_features, train_labels)
    train_bal_acc = balanced_accuracy_score(
        train_labels, clf.predict(train_features))
    val_bal_acc = balanced_accuracy_score(
        val_labels, clf.predict(val_features))

    LOG.info("Training classification report")
    LOG.info(classification_report(train_labels,
             clf.predict(train_features), digits=3))
    LOG.info("train_bal_acc=%.3f", train_bal_acc)
    LOG.info("Validation classification report")
    LOG.info(classification_report(
        val_labels, clf.predict(val_features), digits=3))
    LOG.info("val_bal_acc=%.3f", val_bal_acc)


if __name__ == "__main__":
    from types import SimpleNamespace
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="abstrct", choices=["abstrct", "aae2", "merged"],
                        help="Choose dataset to obtain baseline with")
    parser.add_argument("--extractor", default="distilbert",
                        choices=["distilbert", "scibert", "sbert"],
                        help="Choose the feature extractor")
    args = parser.parse_args()

    configs = dict(seed=42, dataset=args.dataset, batch_size=32, device="cuda",
                   extractor=args.extractor)
    args = SimpleNamespace(**configs)

    try:
        main(args)
    except Exception:
        import ipdb
        ipdb.post_mortem()
