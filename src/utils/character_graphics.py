import matplotlib.pyplot as plt
import numpy as np

from src import character_utils

SIZE=32
shape = (SIZE, SIZE, 1)
#Based on https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb
def plot_images(images, cls_true, cls_pred=None):

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    for i, ax in enumerate(axes.flat):
        # Plot image.

        ax.imshow(images[i].reshape((SIZE, SIZE)), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(character_utils.cls2str(cls_true[i]))
        else:
            xlabel = "True: {0}, Pred: {1}".format(character_utils.cls2str(cls_true[i]), character_utils.cls2str(cls_pred[i]))

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

def plot_weights(weights):
    min_weight = np.min(weights)
    max_weight = np.max(weights)
    fig, axes = plt.subplots(8, 8)
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    for i, ax in enumerate(axes.flat):
        if i < 62:
            heatmap = weights[:, i].reshape(shape)
            ax.set_xlabel("Heatmap {}".format(character_utils.cls2str(i)))
            ax.imshow(heatmap, vmin=min_weight, vmax=max_weight, cmap='seismic')

    ax.set_xticks([])
    ax.set_yticks([])