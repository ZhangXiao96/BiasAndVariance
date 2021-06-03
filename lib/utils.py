import numpy as np


def kl_divergence(x, y, axis=-1, epsilon=1e-10):
    return np.sum(x*np.log(x/(y+epsilon)+epsilon), axis=axis)


def category_2_one_hot(targets, nb_class):
    targets_one_hot = np.zeros(shape=(len(targets), nb_class))
    targets_one_hot[np.arange(len(targets)), targets] = 1.
    return targets_one_hot