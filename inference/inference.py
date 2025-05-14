
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import plot_graph


def N(x):
    return x.detach().cpu().numpy()


def load_model(device):
    """Load model from hub"""
    checkpoint = "david-inf/bert-sci-am"
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model.to(device)
    return model, tokenizer


def inference(dataset, tokenizer, model, device):
    """Load inference dataset and compute argumentative content metrics"""
    metric_per_abstract = []
    for doc in tqdm(dataset, desc="Processing documents", unit="doc"):
        # 0: premise - 1: claim - 2: majclaim (abstrct dataset)
        class_counts = {"0": 0., "1": 0., "2": 0.}
        arg_score = 0.
        for sent in tqdm(doc["sentences"], desc="Processing sentences", leave=False, unit="sent"):
            encoded = tokenizer(sent)
            input_ids = torch.tensor(encoded["input_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoded["attention_mask"]).unsqueeze(0).to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask)  # logits
            # total
            pred = np.argmax(N(output.logits), axis=1)
            logit = np.max(N(output.logits))
            # compute first metric
            class_counts[str(pred.item())] += 1. / len(doc["sentences"])
            # compute second metric
            arg_score += logit / len(doc["sentences"])
        arg_ratio = (class_counts["1"] + class_counts["2"])
        metric_per_abstract.append([arg_ratio, arg_score, doc["label"]])
    return metric_per_abstract  # list of list (len 2)


if __name__ == "__main__":
    device = "cuda:1"
    dataset = load_from_disk("data/inference")
    model, tokenizer = load_model(device)

    m_scores = inference(dataset["molecular"], tokenizer, model, device)
    t_scores = inference(dataset["thoracic"], tokenizer, model, device)

    _, axs = plt.subplots(1, 2, figsize=(10, 6), sharex=True, sharey=True)
    plot_graph(np.array(t_scores), axs[0], "Molecular")
    plot_graph(np.array(m_scores), axs[1], "Thoracic")
    plt.tight_layout()
    plt.savefig("inference/results.svg")
