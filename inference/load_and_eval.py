"""Quick program for loading a model and evaluating on test data"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_dataset
from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import confusion_matrix


def N(x):
    return x.detach().cpu().numpy()


def load_model():
    """Load model from hub"""
    # checkpoint = "david-inf/bert-sci-am"
    checkpoint = "src/ckpts/sbert_full_abstrct"
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return model, tokenizer


def get_loader(opts, tokenizer):
    if opts.dataset == "abstrct":
        dataset = load_dataset("david-inf/am-nlp-mixed")
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
    accs = []
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
            acc = np.mean(pred == N(y))
            accs.append(acc)

    unique, counts = np.unique(trues)
    print(unique, counts)
    print(confusion_matrix(trues, preds))

    return np.mean(accs)


def main(opts):
    model, tokenizer = load_model()
    model.to(opts.device)

    test_loader = get_loader(opts, tokenizer)

    test_acc = test(opts, model, test_loader)
    print("val_acc=%.3f" % test_acc)


if __name__ == "__main__":
    from types import SimpleNamespace
    configs = {"device": "cuda", "dataset": "abstrct",
               "split_name": "validation"}
    opts = SimpleNamespace(**configs)
    try:
        main(opts)
    except Exception as e:
        print(e)
