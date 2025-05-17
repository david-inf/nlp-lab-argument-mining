
import numpy as np
import matplotlib.pyplot as plt


def plot_graph(vals_mat, ax, title):
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
    ax.set_xlabel("argument ratio")
    ax.set_ylabel("argument score")
    ax.legend()
    ax.grid(True)
