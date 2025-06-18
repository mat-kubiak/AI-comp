import torch
import numpy as np

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from captum.attr import IntegratedGradients
from captum.attr import Lime
from captum._utils.models.linear_model import SkLearnLinearRegression

from skimage.segmentation import slic

from src.cnn_imagenette import CNN
from src.utils import plot_2_part_heatmap_rgb

NUM_SAMPLES = 3

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
    ])

    dataset = datasets.Imagenette(root='datasets', split='val', size="320px", transform=transform, download=True)

    model = CNN.load_from_checkpoint('checkpoints/imagenette_cnn_best_71.ckpt')
    model.to(device)
    model.eval()

    # Unpack data
    X, y = [], []
    for i in range(len(dataset)):
        xi, yi = dataset[i]  # this applies the transform!
        X.append(xi.numpy())  # assumes transform outputs a tensor
        y.append(yi)

    X = np.array(X)
    y = np.array(y)

    # Choose sample indices
    random_indices = np.random.choice(len(X), size=NUM_SAMPLES, replace=False)
    samples = torch.tensor(X[random_indices], dtype=torch.float32).to(device)
    labels = torch.tensor(y[random_indices]).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(samples)
        preds = torch.argmax(outputs, dim=1)
    print("Random sample indices:", random_indices)
    print("Predictions: ", preds.cpu().numpy())
    print("Ground truth:", labels.cpu().numpy())

    # Integrated Gradients
    print('\n### Integrated Gradents ###############\n')
    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(samples, target=labels, return_convergence_delta=True)

    attributions = np.reshape(attributions.cpu(), (NUM_SAMPLES, 3, 160, 160))
    for i in range(len(attributions)):
        plot_2_part_heatmap_rgb(np.reshape(samples[i].cpu(), (3, 160, 160)), attributions[i], f'Integrated Gradients\ny_true: {labels[i]}, y_pred: {preds[i]}')

if __name__ == '__main__':
    main()
