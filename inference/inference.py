
import os

import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
import matplotlib.pyplot as plt
from tqdm import tqdm

from inf_utils import N, plot_graph, argscore_argratio, LOG


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
    metrics_per_abstract = []  # list of list (len 3)
    with torch.no_grad():
        for doc in tqdm(dataset, desc="Documents", unit="doc"):
            # 0: premise - 1: claim - 2: majclaim (abstrct+sciarg dataset)

            encoded_doc = tokenizer(
                doc["sentences"],  # list of str
                max_length=128,  # max sentence length
                truncation=True,  # truncate if exceeds max_length
                # all sentences are padded to min(128, longest_seq)
                padding="longest",
                return_attention_mask=True,  # avoids padding tokens
                return_tensors="pt",  # return torch.Tensor
            ).to(device)

            input_ids = encoded_doc["input_ids"]
            attention_mask = encoded_doc["attention_mask"]
            label = doc["label"]  # int
            # output.logits -> [N_i, 3]
            output = model(input_ids=input_ids, attention_mask=attention_mask)

            metrics = argscore_argratio(N(output.logits), metric_id)
            arg_ratio, arg_score = metrics.values()
            xlab, ylab = metrics.keys()

            # update metrics for this document
            # LOG.info("Document stats: sentences=%s, classes=%s, label=%s",
            #         input_ids.size(0), class_counts, label)
            # LOG.info("Metrics: AR=%.3f, AS=%.3f, total_logits=%.3f, claim_logits=%.3f",
            #         arg_ratio, arg_score, sum_of_logits, sum_of_claim_logits)
            metrics_per_abstract.append([arg_ratio, arg_score, label])

    return metrics_per_abstract, xlab, ylab


def main(opts):
    """Launch inference"""
    # Load inference dataset (Molecular and Thoracic splits)
    dataset = load_from_disk("data/inference")
    # Load model and its tokenizer
    model, tokenizer = load_model(opts.checkpoint, opts.device)

    # Inference on Molecular split
    molecular_scores, xlab, ylab = inference(
        dataset["molecular"], model, tokenizer, opts.device, opts.metric_id)
    # Inference on Thoracic split
    thoracic_scores, xlab, ylab = inference(
        dataset["thoracic"], model, tokenizer, opts.device, opts.metric_id)

    # Plot results
    _, axs = plt.subplots(1, 2, figsize=(10, 6), sharex=True, sharey=True)
    plot_graph(np.array(molecular_scores), axs[0], "Molecular", xlab, ylab)
    plot_graph(np.array(thoracic_scores), axs[1], "Thoracic", xlab, ylab)

    plt.tight_layout()
    output_path = os.path.join(
        "inference/results", f"plot_{opts.metric_id}.svg")
    LOG.info("output_path=%s", {output_path})
    plt.savefig(output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1",
                        help="Choose compute device")
    parser.add_argument("--checkpoint", "-c", default="david-inf/bert-sci-am",
                        help="Provide model checkpoint")
    parser.add_argument("--metric_id", "-m", type=int,
                        help="Specify an id for the metrics pair")

    # checkpoint = "/data01/dl24davnar/projects/nlp-lab-argument-mining/src/ckpts/sbert_full_abstrct"
    # checkpoint = "/data01/dl24davnar/projects/nlp-lab-argument-mining/src/ckpts/sbert_full_sciarg"
    # checkpoint = "/data01/dl24davnar/projects/nlp-lab-argument-mining/src/ckpts/sbert_full_mixed"

    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        import ipdb
        import traceback
        import sys
        print("Exception:", e)
        traceback.print_exc()
        ipdb.post_mortem(sys.exc_info()[2])
