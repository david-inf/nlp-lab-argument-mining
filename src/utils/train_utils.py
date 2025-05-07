
import numpy as np


def accuracy(logits, labels):
    """Compute accuracy during pytorch training"""
    pred = np.argmax(logits, axis=1)
    acc = np.mean(pred == labels)
    return acc


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        """Initialize the AverageMeter with default values."""
        # store metric statistics
        self.val = 0  # value
        self.sum = 0  # running sum
        self.avg = 0  # running average
        self.count = 0  # steps counter

    def reset(self):
        """Reset all statistics to zero."""
        # store metric statistics
        self.val = 0  # value
        self.sum = 0  # running sum
        self.avg = 0  # running average
        self.count = 0  # steps counter

    def update(self, val, n=1):
        """Update statistics with new value.
        
        Args:
            val: The value to update with
            n: Weight of the value (default: 1)
        """
        # update statistic with given new value
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# TODO: early stopping class
