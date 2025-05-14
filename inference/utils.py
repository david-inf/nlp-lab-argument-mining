
import numpy as np
import matplotlib.pyplot as plt


def plot_graph(vals_mat, ax, title):
    # Create scatter plot with two colors based on vals_mat[:, 2]
    is_non_argument = vals_mat[:, 2] == 0.
    
    # Plot excluded (E)
    ax.scatter(
        vals_mat[is_non_argument, 0],
        vals_mat[is_non_argument, 1],
        c='red',
        label='E'
    )

    # Plot accepted (A)
    ax.scatter(
        vals_mat[~is_non_argument, 0],
        vals_mat[~is_non_argument, 1],
        c='blue',
        label='A'
    )

    ax.set_title(title)
    ax.set_xlabel("argument ratio")
    ax.set_ylabel("argument score")
    ax.legend()
    ax.grid(True)

