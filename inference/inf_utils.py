
import logging
import torch
import numpy as np
import pandas as pd
from rich.logging import RichHandler


def N(x: torch.Tensor):
    return x.detach().cpu().numpy()


def get_logger():
    """Logging mechanism"""
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    return log


LOG = get_logger()


def plot_graph(vals_mat, ax, title, xlab, ylab):
    # vals_max = [n_abstracts, AS, AR, 0/1]
    # Create scatter plot with two colors based on vals_mat[:, 2]
    excluded_idx = vals_mat[:, 2] == 0.

    # Plot excluded (E)
    ax.scatter(
        vals_mat[excluded_idx, 0],
        vals_mat[excluded_idx, 1],
        c='red',
        label='E'
    )

    # Plot accepted (A)
    ax.scatter(
        vals_mat[~excluded_idx, 0],
        vals_mat[~excluded_idx, 1],
        c='blue',
        label='A'
    )

    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.legend()
    ax.grid(True)


def mixed_scores(logits):
    """
    Scores based on the Mixed dataset training for a single article

    Args
        logits: np.ndarray of shape (N_i, 3) for a single document
    """
    # predictions for current document sentences
    pred = np.argmax(logits, axis=1)  # (N_i,), class prediction

    class_0 = np.where(pred == 0, 1, 0).sum()  # counts for class Premise
    class_1 = np.where(pred == 1, 1, 0).sum()  # counts for class Claim
    class_2 = np.where(pred == 2, 1, 0).sum()  # counts for class MajorClaim
    class_counts = {"0": class_0, "1": class_1, "2": class_2}

    num_sents = logits.shape[0]  # number of sentences in the current document
    num_claims = class_counts["1"] + class_counts["2"]  # number of claims
    num_premises = class_counts["0"]

    prem_ratio = num_premises / num_sents
    claim_ratio = num_claims / num_sents

    topk_claim = torch.topk(
        torch.from_numpy(logits[:, 1:]), 10, dim=0).values.sum().numpy() / 10
    avg_claim_score = logits[:, 1:].max(axis=1).sum() / num_sents
    avg_prem_score = logits[:, 0].sum() / num_sents

    return [prem_ratio, claim_ratio, topk_claim, avg_claim_score, avg_prem_score]


def ibm_scores(logits):
    """
    Scores based on the IBM dataset training for a single article

    Args
        logits: np.ndarray of shape (N_i, 3) for a single document
    """
    # predictions for current document sentences
    pred = np.argmax(logits, axis=1)  # (N_i,), class prediction

    class_0 = np.where(pred == 0, 1, 0).sum()  # counts for class Other
    class_1 = np.where(pred == 1, 1, 0).sum()  # counts for class Evidence
    class_2 = np.where(pred == 2, 1, 0).sum()  # counts for class Claim
    class_counts = {"0": class_0, "1": class_1, "2": class_2}

    # num_sents = logits.shape[0]  # number of sentences in the current document
    # num_claims = class_counts["1"] + class_counts["2"]  # number of claims
    # num_premises = class_counts["0"]

    # prem_ratio = num_premises / num_sents
    # claim_ratio = num_claims / num_sents

    # topk_claim = torch.topk(
    #     torch.from_numpy(logits[:, 1:]), 10, dim=0).values.sum().numpy() / 10
    # avg_claim_score = logits[:, 1:].max(axis=1).sum() / num_sents
    # avg_prem_score = logits[:, 0].sum() / num_sents

    return [prem_ratio, claim_ratio, topk_claim, avg_claim_score, avg_prem_score]
