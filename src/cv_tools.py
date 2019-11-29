import numpy as np
import pandas as pd
from math import ceil
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_images_grid(images, labels=("image", "label"), categories=None,
                      grid_side=10, fig_height=15, fig_width=15,
                      text_label_width=100, text_label_height=35):
    if not isinstance(images, pd.DataFrame):
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


def image_generator():
    return ImageDataGenerator(
        featurewise_center=False,             # set input mean to 0 over the dataset
        samplewise_center=False,              # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,   # divide each input by its std
        zca_whitening=False,                  # apply ZCA whitening
        rotation_range=360,                   # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.15,                    # Randomly zoom image
        width_shift_range=0.15,               # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.15,              # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,                 # randomly flip images
        vertical_flip=True)                   # randomly flip images


def equalizer_augmentation(images, labels, label, batch=64):
    total_count = labels[label].value_counts(sort=True)
    higher_count = total_count.values[0]

    fill_info = {}
    for idx, count in zip(total_count.index[1:], total_count.values[1:]):
        fill_info[idx] = higher_count - count

    images_generated = None
    labels_generated = None
    datagen = image_generator()
    datagen.fit(images)

    for key, val in fill_info.items():
        print(f"Generating {val} images for label {key}")
        selection = labels[labels['class'] == key]
        ngen = val if val <= batch else batch
        gen = datagen.flow(images[selection.index], np.full(selection.shape[0], key), batch_size=ngen)

        counter = 0
        while counter < val:
            if val - counter < batch:
                gen = datagen.flow(images[selection.index],
                                   np.full(selection.shape[0], key),
                                   batch_size=(val - counter))

            new_images, new_labels = next(gen)
            if images_generated is None:
                images_generated = new_images
                labels_generated = new_labels
            else:
                images_generated = np.concatenate([images_generated, new_images], axis=0)
                labels_generated = np.concatenate([labels_generated, new_labels])
            counter += batch

    total_images = np.concatenate([images, images_generated], axis=0)
    total_labels = np.concatenate([labels['class'].values, labels_generated], axis=0)
    return total_images, total_labels


def plot_filters(model, layer=None, ncols=6):
    layer_names = [(idx, k.name) for idx, k in enumerate(model.layers) if "conv2d" in k.name]
    if layer is None:
        idx, layer = layer_names[-1]
    else:
        check_names = [(idx, k.name) for idx, k in enumerate(model.layers) if layer in k.name]
        if len(check_names) > 0:
            idx, layer = check_names[0]
        else:
            raise ValueError(f"The layer {layer} was not found")
    try:
        nrows = ceil(model.layers[idx].get_weights()[0].shape[-1] / ncols)
    except:
        nrows = ceil(model.layers[idx].get_weights().shape[-1] / ncols)

    i = 0
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            ax[row, col].imshow(model.layers[idx].get_weights()[0][:, :, row, col])
            ax[row, col].set_title(
                f"fltr:{i}")
            i += 1
    fig.set_figheight(8)
    fig.set_figwidth(15)


def display_features(features, index, col_size, row_size=None,
                     fig_height=8, fig_width=15, cmap=None, limit=300):
    """
    Plot the features of a tensorflow activation model
    :param features:
    :param index:
    :param col_size:
    :param row_size:
    :param fig_height:
    :param fig_width:
    :param cmap: e.x. 'gray'
    :return:
    """
    feature = features[index]
    max_idx = feature.shape[-1]
    if row_size is None:
        row_size = ceil(max_idx / col_size)
    if cmap is None:
        fig_params = {}
    else:
        fig_params = {"cmap": cmap}

    feature_index = 0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5, col_size*1.5))
    for row in range(0, row_size):
        if feature_index == max_idx:
            break
        for col in range(0, col_size):
            imag = feature[0, :, :, feature_index]
            ax[row][col].imshow(imag, **fig_params)
            feature_index += 1

    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)
