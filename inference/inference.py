"""Compute scores per each article and save to CSV"""

import os

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm

from inf_utils import N, LOG, mixed_scores, ibm_scores


def load_model(checkpoint, device):
    """Load model from hub"""
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-mpnet-base-v2")
    model.to(device)
    return model, tokenizer


def inference(opts, dataset, model, tokenizer, device):
    """Load inference dataset and compute argumentative content metrics"""
    model.eval()
    metrics_per_article = []  # list of lists
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

            if opts.ftdata == "mixed":
                scores = mixed_scores(N(output.logits))
            else:
                scores = ibm_scores(N(output.logits))
            scores.append(label)
            # scores = [p1, p2, p3,..., label] i.e. a train sample
            metrics_per_article.append(scores)
            # free memory
            del encoded_doc, input_ids, attention_mask, output
            torch.cuda.empty_cache()

    df = pd.DataFrame(
        np.array(metrics_per_article),
        columns=["PR", "CR", "topk", "ACS", "APS", "C1", "C2", "C3", "label"])

    return df


def main(opts):
    """Launch inference"""
    # Load inference dataset (Molecular and Thoracic splits)
    dataset = load_from_disk("data/inference")
    # Load model and its tokenizer
    model, tokenizer = load_model(opts.checkpoint, opts.device)

    output_dir = "inference/scores"
    os.makedirs(output_dir, exist_ok=True)

    # Inference on Molecular split
    molecular_scores_df = inference(
        opts, dataset["molecular"], model, tokenizer, opts.device)

    molecular_path = os.path.join(
        output_dir, f"molecular_{opts.ftdata}.csv")
    molecular_scores_df.to_csv(
        molecular_path, index=False)
    LOG.info("Molecular scores saved to CSV at path=%s", {molecular_path})

    # Inference on Thoracic split
    thoracic_scores_df = inference(
        opts, dataset["thoracic"], model, tokenizer, opts.device)

    thoracic_path = os.path.join(
        output_dir, f"thoracic_{opts.ftdata}.csv")
    thoracic_scores_df.to_csv(
        thoracic_path, index=False)
    LOG.info("Saving thoracic scores to path=%s", {thoracic_path})

    # Molecular + Thoracic
    merge_scores_df = pd.concat(
        [molecular_scores_df, thoracic_scores_df], ignore_index=True)
    merge_path = os.path.join(
        output_dir, f"merge_{opts.ftdata}.csv")
    merge_scores_df.to_csv(merge_path, index=False)
    LOG.info("Merged scores saved to path=%s", {merge_path})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1",
                        help="Choose compute device")
    parser.add_argument("--ftdata", choices=["mixed", "ibm"],
                        default="mixed",
                        help="Choose finetuned bert")
    # parser.add_argument("--checkpoint", "-c", default="david-inf/bert-sci-am",
    #                     help="Provide model checkpoint")
    args = parser.parse_args()

    if args.ftdata == "mixed":
        # checkpoint = "/data01/dl24davnar/projects/nlp-lab-argument-mining/src/ckpts/distilbert_full_mixed"
        checkpoint = "/data01/dl24davnar/projects/nlp-lab-argument-mining/src/ckpts/sbert_full_mixed"
    else:
        checkpoint = "/data01/dl24davnar/projects/nlp-lab-argument-mining/src/ckpts/sbert_full_ibm"
    args.checkpoint = checkpoint

    try:
        main(args)
    except Exception as e:
        import ipdb
        import traceback
        import sys
        print("Exception:", e)
        traceback.print_exc()
        ipdb.post_mortem(sys.exc_info()[2])
