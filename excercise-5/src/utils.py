import numpy as np
import matplotlib.pyplot as plt

def plot_heatmap(data, title=''):
    max_val = np.abs(data).max()

    fig, ax = plt.subplots()
    im = ax.imshow(data, vmin=-max_val, vmax=max_val, cmap='bwr')
    if title != '':
        ax.set_title(title)
    fig.colorbar(im, ax=ax)
    plt.show()

def collect_samples_per_class(x, y_true, y_pred, per_class=50, only_correct=True):
    """
        Choose `per_class` number of samples for every class by random.
        You can ensure only correctly guessed samples are taken into account by settng `only_correct` to True.
    """
    idx_by_class = {i: [] for i in range(10)}

    # sort dataset into classes
    for idx in range(len(x)):
        label = y_true[idx]
        if only_correct and y_pred[idx] != label:
            continue
        idx_by_class[label].append(idx)

    # sample randomly
    x_by_class = {}
    y_by_class = {}
    for cls in range(10):
        selected = np.random.choice(idx_by_class[cls], per_class, replace=False)

        x_by_class[cls] = [x[i] for i in selected]
        y_by_class[cls] = [y_pred[i].item() for i in selected]
        idx_by_class[cls] = selected

    return idx_by_class, x_by_class, y_by_class

def shuffle_by_value(data):
    """
        shuffles unique values of an array
        (i.e. all 3s become 6s and all 6s become 3s)
    """
    unique_data = np.unique(data)
    shuffled_labels = np.random.permutation(unique_data)

    label_map = dict(zip(unique_data, shuffled_labels))

    return np.vectorize(label_map.get)(data)

def plot_2_part_heatmap(original_data, attr_data, title=''):
    """
    plot original image and a heatmap
    """
    max_val = np.abs(attr_data).max()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Original image
    axes[0].imshow(original_data, cmap='gray')
    axes[0].set_title('Original Image')

    # Attribution heatmap
    im = axes[1].imshow(attr_data, vmin=-max_val, vmax=max_val, cmap='bwr')
    axes[1].set_title('Attributions')
    fig.colorbar(im, ax=axes[1])

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()

def plot_3_part_heatmap(original_data, segment_data, attr_data, title=''):
    """
    plot original image, SLIC segmentation and a heatmap
    """
    max_val = np.abs(attr_data).max()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Original image
    axes[0].imshow(original_data, cmap='gray')
    axes[0].set_title('Original Image')

    # Segmentation map
    shuffled_segments = shuffle_by_value(segment_data)
    axes[1].imshow(shuffled_segments, cmap='nipy_spectral')
    axes[1].set_title('Segmentation')

    # Attribution heatmap
    im = axes[2].imshow(attr_data, vmin=-max_val, vmax=max_val, cmap='bwr')
    axes[2].set_title('Attributions')
    fig.colorbar(im, ax=axes[2])

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()
