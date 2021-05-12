from matplotlib import pyplot as plt
from matplotlib import axes, patches, path
from torch import Tensor
from data import disc_radius, disc_center

def plot_dataset(
    dataset: tuple[Tensor, Tensor],
    ax: axes.Axes,
    cmap: dict = {0: "#2a10d1", 1: "#d17e10"},
    plot_boundary: bool = True
):
    for coordinate, label in zip(*dataset):
        x, y = coordinate
        color = cmap[label.item()]
        marker = {0: "o", 1: "+"}[label.item()]
        if plot_boundary:
            patch = patches.Circle(disc_center, disc_radius, fill=False, ls='--')
            ax.add_patch(patch)
        ax.scatter(x, y, c=color, marker=marker)
        ax.set_aspect('equal')


def plot_results(
    train_data: tuple[Tensor, Tensor],
    test_data: tuple[Tensor, Tensor],
    correct_class: Tensor
):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21,7), subplot_kw=dict(box_aspect=1))

    ax1.set_title('Training data')
    plot_dataset(train_data, ax1)

    ax2.set_title('Test data')
    plot_dataset(test_data, ax2)

    ax3.set_title('Test data prediction accuracy')
    plot_dataset((test_data[0], correct_class.int()), ax3, cmap={0: '#ff0000', 1: '#00ff00'}, plot_boundary=False)
    plt.savefig('plots/test-results')
    plt.show()
