
import os

import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import plot_graph, LOG, argscore_argratio


def N(x):
    return x.detach().cpu().numpy()


def load_model(checkpoint, device):
    """Load model from hub"""
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2")
    model.to(device)
    return model, tokenizer


def inference(dataset, model, tokenizer, device, metric_id):
    """Load inference dataset and compute argumentative content metrics"""
    model.eval()
    metrics_per_abstract = []
    with torch.no_grad():
        for doc in tqdm(dataset, desc="Documents", unit="doc"):
            # 0: premise - 1: claim - 2: majclaim (abstrct+sciarg dataset)

            encoded_doc = tokenizer(
                doc["sentences"],  # list of str
                max_length=128,  # max sentence length
                truncation=True,  # truncate if exceeds max_length
                padding="longest",  # all sentences are padded to reach 128 tokens
                return_attention_mask=True,  # avoids padding tokens
                return_tensors="pt",  # return torch.Tensor
            ).to(device)

            input_ids = encoded_doc["input_ids"]
            attention_mask = encoded_doc["attention_mask"]
            label = doc["label"]  # int
            # output.logits -> [N_i, 3]
            output = model(input_ids=input_ids, attention_mask=attention_mask)

            # predicition and its logit
            # scalar, sum logits for claims and premises
            # sum_of_logits = N(output.logits).sum()
            # max_claim_logits = N(output.logits)[:, 1:].max(axis=1)  # [2]
            # sum_of_claim_logits = max_claim_logits.sum()  # sum logits for claims

            # compute argumentative score
            # arg_score = sum_of_claim_logits / sum_of_logits
            # arg_score = sum_of_claim_logits / input_ids.size(0)
            # arg_score = np.max(N(output.logits)) / input_ids.size(0)
            # arg_score = torch.topk(torch.from_numpy(
            #     max_claim_logits), 10).values.sum().numpy() / input_ids.size(0)
            
            metrics = argscore_argratio(N(output.logits), metric_id)
            arg_ratio, arg_score = metrics.values()
            xlab, ylab = metrics.keys()

            # update metrics for this document
            # LOG.info("Document stats: sentences=%s, classes=%s, label=%s",
            #         input_ids.size(0), class_counts, label)
            # LOG.info("Metrics: AR=%.3f, AS=%.3f, total_logits=%.3f, claim_logits=%.3f",
            #         arg_ratio, arg_score, sum_of_logits, sum_of_claim_logits)
            metrics_per_abstract.append([arg_ratio, arg_score, label])

    return metrics_per_abstract, xlab, ylab  # list of list (len 2)


# def main(opts):


if __name__ == "__main__":
    device = "cuda:1"
    dataset = load_from_disk("data/inference")
    # checkpoint = "/data01/dl24davnar/projects/nlp-lab-argument-mining/src/ckpts/sbert_full_abstrct"
    # checkpoint = "/data01/dl24davnar/projects/nlp-lab-argument-mining/src/ckpts/sbert_full_sciarg"
    # checkpoint = "/data01/dl24davnar/projects/nlp-lab-argument-mining/src/ckpts/sbert_full_mixed"
    checkpoint = "david-inf/bert-sci-am"
    model, tokenizer = load_model(checkpoint, device)

    metric_id = 2
    # m_loader = get_loader(dataset["molecular"], tokenizer)
    # m_scores = inference(m_loader, model, device)
    molecular_scores, xlab, ylab = inference(
        dataset["molecular"], model, tokenizer, device, metric_id)

    # s_loader = get_loader(dataset["thoracic"], tokenizer)
    # t_scores = inference(s_loader, model, device)
    thoracic_scores, xlab, ylab = inference(
        dataset["thoracic"], model, tokenizer, device, metric_id)

    _, axs = plt.subplots(1, 2, figsize=(10, 6), sharex=True, sharey=True)
    plot_graph(np.array(molecular_scores), axs[0], "Molecular", xlab, ylab)
    plot_graph(np.array(thoracic_scores), axs[1], "Thoracic", xlab, ylab)

    plt.tight_layout()
    output_path = os.path.join("inference", f"plot_{metric_id}.svg")
    plt.savefig(output_path)
