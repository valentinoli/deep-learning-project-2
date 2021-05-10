from __future__ import annotations
import math
from torch import empty, Tensor


disc_radius = 1/math.sqrt(2 * math.pi)
disc_center = (.5, .5)

def compute_label(point: tuple[float, float]) -> int:
    """
    :param point: tuple (x, y) representing a point in the Euclidean plane
    :returns: 1 if point is inside the disc centered at (0.5, 0.5) of radius 1/sqrt(2*pi), 0 otherwise
    """
    return int(math.dist(point, disc_center) <= disc_radius)

def generate_samples(n: int) -> tuple[Tensor, Tensor]:
    """
    :param n: number of samples
    :returns:
        inputs - n points sampled uniformly from [0,1]^2
        labels - n one-hot labels (see function compute_label)
    """
    inputs = empty(n, 2).uniform_()
    labels = empty(n).new_tensor(list(map(compute_label, inputs))).view(-1, 1)
    return inputs, labels

def generate_data(n: int = 1000) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    """
    :param n: number of samples
    :returns:
        train_inputs - n training points
        train_labels - n training labels
        test_inputs - n test points
        test_labels - n test labels
    """
    return generate_samples(n), generate_samples(n)


def create_batches(inputs: Tensor, labels: Tensor, size: int) -> list[tuple[Tensor, Tensor]]:
    """
    :param inputs: input tensor
    :param labels: label tensor
    :param size: batch size
    :returns: minibatches of inputs and labels
    """
    assert len(inputs) == len(labels)
    return [(inputs[i:i+size], labels[i:i+size]) for i in range(0, len(inputs), size)]


class DataLoader():
    """Data loader class for iterating over minibatches"""
    def __init__(self, data: tuple[Tensor, Tensor], batch_size: Optional[int] = 10):
        self.batches = create_batches(*data, batch_size)
    def __iter__(self):
        self.iter = iter(self.batches)
        return self.iter
    def __next__(self):
        return next(self.iter)
