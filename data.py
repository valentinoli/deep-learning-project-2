import math
from typing import Union

from torch import empty, Tensor


disc_radius = 1/math.sqrt(2 * math.pi)
disc_center = (.5, .5)

def compute_label(point: tuple[float, float]) -> int:
    """
    :param point: tuple (x, y) representing a point in the Euclidean plane
    :returns: 1 if point is inside the disc centered at (0.5, 0.5) of radius 1/sqrt(2*pi), 0 otherwise
    """
    return int(math.dist(point, disc_center) <= disc_radius)

def generate_samples(n) -> tuple[Tensor, Tensor]:
    """
    :param n: number of samples
    :returns:
        inputs - n points sampled uniformly from [0,1]^2
        labels - n one-hot labels (see function compute_label)
    """
    inputs = empty(n, 2).uniform_()
    labels = empty(n).new_tensor(list(map(compute_label, inputs))).view(-1, 1)
    return inputs, labels

def generate_data(n: int = 1000) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    :param n: number of samples
    :returns:
        train_inputs - n training points
        train_labels - n training labels
        test_inputs - n test points
        test_labels - n test labels
    """
    return (*generate_samples(n), *generate_samples(n))
