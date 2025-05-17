
import numpy as np
import matplotlib.pyplot as plt
import logging
from rich.logging import RichHandler


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


# def argscore_argratio()
