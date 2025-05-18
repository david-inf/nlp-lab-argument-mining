
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
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


def argscore_argratio(logits, metric_id):
    """
    Argumentative score and ratio can be defined in many ways

    Args
        logits: np.ndarray of shape (N_i, 3) for a single document
        metric_id: id of the metrics to use
    """
    # predictions for current document sentences
    pred = np.argmax(logits, axis=1)  # (N_i,), class prediction

    class_0 = np.where(pred == 0, 1, 0).sum()  # counts for class 0
    class_1 = np.where(pred == 1, 1, 0).sum()  # counts for class 1
    class_2 = np.where(pred == 2, 1, 0).sum()  # counts for class 2
    class_counts = {"0": class_0, "1": class_1, "2": class_2}

    num_sents = logits.shape[0]  # number of sentences in the current document
    num_claims = class_counts["1"] + class_counts["2"]  # number of claims
    num_premises = class_counts["0"]

    if metric_id == 0:
        xlab = "PR (fraction of premises)"
        ylab = "AS (max claim score divided by logits)"
        # fraction of sentences predicted as premises (class 0)
        xmetric = num_premises / num_sents
        # 
        sum_of_logits = logits[:, 1:].sum()  # scalar
        max_claim_logits = logits[:, 1:].max(axis=1)  # [N]
        ymetric = max_claim_logits.sum() / sum_of_logits  # scalar

    elif metric_id == 1:
        xlab = "PR (fraction of premises)"
        ylab = "AS (mean max claim score)"
        # number of sentences predicted as claim (class 1 or 2)
        xmetric = num_premises / num_sents
        # 
        max_claim_logits = logits[:, 1:].max(axis=1)  # [N]
        ymetric = max_claim_logits.sum() / num_sents

    elif metric_id == 2:
        xlab = "AR (number of claims)"
        ylab = "AS (topk claim score)"
        # number of claims
        xmetric = num_claims
        # topk scores
        k = 10
        topk = torch.topk(torch.from_numpy(logits[:, 1:]), k, dim=0).values
        ymetric = topk.sum().numpy() / (k*2)

    elif metric_id == 3:
        xlab = "AR (fraction of claims)"
        ylab = "AS (mean claim score)"
        # fraction of sentences predicted as claim (class 1 or 2)
        xmetric = num_claims / num_sents
        # mean logit for claim classes
        # (mean class 1 + mean class 2) / 2
        ymetric = logits[:, 1:].sum() / (num_sents*2)

    elif metric_id == 4:
        xlab = "AR (fraction of claims)"
        ylab = "AS (topk claim score)"
        # fraction of sentences predicted as claim (class 1 or 2)
        xmetric = num_claims / num_sents
        # topk argumentative score (compute as then choose topk)
        k = 10
        topk = torch.topk(torch.from_numpy(logits[:, 1:]), k, dim=0).values
        ymetric = topk.sum().numpy() / (k*2)

    # elif metric_id == 4:
    #     xlab = "AR (fraction of claims)"
    #     ylab = "AS (scores on scores)"

    else:
        raise ValueError(f"Unknown metric {metric_id}")

    return {xlab: xmetric, ylab: ymetric}
