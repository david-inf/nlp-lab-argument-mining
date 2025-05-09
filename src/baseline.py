"""Baseline to improve with finetuning"""

import numpy as np

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, balanced_accuracy_score

from datasets import load_from_disk
from transformers import pipeline

from utils.misc_utils import set_seeds, LOG


def get_dataset(opts):
    """Dataset splits"""
    if opts.dataset == "abstrct":
        dataset = load_from_disk("data/finetuning/abstrct")
    elif opts.dataset == "aae2":
        dataset = load_from_disk("data/finetuning/aae2")
    elif opts.dataset == "merged":
        dataset = load_from_disk("data/finetuning/merged")
    else:
        raise ValueError(f"Unknown dataset {opts.dataset}")
    return dataset["train"], dataset["validation"]


def bert_features(texts, bert_name):
    """Use DistilBERT/SciBERT as feature extractor"""
    if bert_name == "distilbert":
        checkpoint = "distilbert-base-uncased"
    elif bert_name == "scibert":
        checkpoint = "allenai/scibert_scivocab_uncased"
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


# TODO: sbert_features()


def main(opts):
    """Extract features and train classifier"""
    set_seeds(opts.seed)
    # Get train-val split
    trainset, valset = get_dataset(opts)

    train_labels = np.array(trainset["label"])
    val_labels = np.array(valset["label"])
    # Extract features
    if opts.extractor in ("distilbert", "scibert"):
        train_features = bert_features(trainset["text"], opts.extractor)
        val_features = bert_features(valset["text"], opts.extractor)
    # TODO: SBERT and other integrations
    else:
        raise ValueError(f"Unknown extractor {opts.extractor}")

    # Train classifier and do inference
    if opts.classifier == "svm":
        LOG.info("LinearSVC classifier")
        clf = LinearSVC()
    elif opts.classifier == "logistic":
        LOG.info("LogisticRegression classifier")
        clf = LogisticRegression()
    else:
        raise ValueError(f"Unknown classifier {opts.classifier}")

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
    parser.add_argument("--extractor", default="distilbert", choices=["distilbert", "scibert"],
                        help="Choose the feature extractor")
    parser.add_argument("--classifier", default="svm", choices=["svm", "logistic"],
                        help="Classifier to train ontop of DistilBERT features")
    args = parser.parse_args()

    configs = dict(seed=42, dataset=args.dataset, batch_size=32, device="cuda",
                   extractor=args.extractor, classifier=args.classifier)
    args = SimpleNamespace(**configs)

    try:
        main(args)
    except Exception:
        import ipdb
        ipdb.post_mortem()
