import numpy as np
import matplotlib.pyplot as plt

from src.loaders import cifar100_loader as cf100
from src.loaders.class_names import CLASS_NAMES


def summarise_set(name, images, labels):
    print(f"{name}: {images.shape[0]} images, image shape {images.shape[1:]}, dtype {images.dtype}")
    print(f"{name}: {len(np.unique(labels))} classes -> {', '.join(CLASS_NAMES)}")


def show_training_samples(num_classes, X_train, y_train):
    num_samples = []
    cols = 5

    fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(6, 28))
    fig.tight_layout()
    for i in range(cols):
        for j, name in enumerate(CLASS_NAMES):
            x_selected = X_train[y_train == j]
            axs[j][i].imshow(x_selected[np.random.randint(0, len(x_selected) - 1), :, :], cmap=plt.get_cmap('gray'))
            axs[j][i].axis("off")
            if i == 2:
                axs[j][i].set_title(str(j) + " - " + name)
                num_samples.append(len(x_selected))
    plt.show()
    return num_samples


def plot_sample_distribution(num_classes, num_of_each_sign):
    plt.figure(figsize=(12, 4))
    plt.bar(range(0, num_classes), num_of_each_sign)
    plt.xticks(range(0, num_classes), CLASS_NAMES, rotation=90)
    plt.title("Distribution of the training set")
    plt.xlabel("Class number")
    plt.ylabel("Number of images")
    plt.tight_layout()
    plt.show()
