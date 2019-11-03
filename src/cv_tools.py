import numpy as np
import pandas as pd
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
