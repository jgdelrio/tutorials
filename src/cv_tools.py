from pandas import DataFrame
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def build_images_grid(images, labels=("image", "label"), categories=None,
                      grid_side=10, fig_height=15, fig_width=15,
                      text_label_width=100, text_label_height=35):
    if not isinstance(images, DataFrame):
        raise TypeError("The 'images' input must be a pandas DataFrame")
    fig = plt.figure(1, figsize=(grid_side, grid_side))
    grid = ImageGrid(fig, 111, nrows_ncols=(grid_side, grid_side), axes_pad=0.05)
    i = 0

    if categories is None and isinstance(labels, tuple):
        # we have a dataset with at least one column being the images and one column being the labels
        img_ref, label_ref = labels[:2]
        categories = images[label_ref].unique()
        for cat in categories:
            for img in images[images[label_ref] == cat][img_ref][:grid_side]:
                ax = grid[i]
                ax.imshow(img)
                ax.axis('off')
                if i % grid_side == grid_side - 1:
                    ax.text(text_label_width, text_label_height, cat, verticalalignment='center')
                i += 1

    else:
        # we have two datasets labels being one with just one column, the categories
        # and categories is the list of categories that we have selected to plot
        for cat in categories:
            for img in images[labels.iloc[:, 0] == cat][:grid_side]:
                ax = grid[i]
                ax.imshow(img)
                ax.axis('off')
                if i % grid_side == grid_side - 1:
                    ax.text(text_label_width, text_label_height, cat, verticalalignment='center')
                i += 1

    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)
