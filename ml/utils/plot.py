import matplotlib.pyplot as plt


def plot_matrix(matrix, output_image_path, title=None):
    """
    Plot relative heart rate errors histogram.

    :param matrix: 2d numpy array
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if title is not None:
        ax.set_title(title)
    plt.imshow(matrix)
    plt.colorbar()
    plt.savefig(output_image_path)
    plt.close(fig)
