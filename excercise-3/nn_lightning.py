import torch
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, silhouette_score, \
    pairwise_distances

from config import LEARNING_RATE, INPUT_SIZE, NUM_CLASSES, HIDDEN_LAYERS, BATCH_SIZE, EPOCHS, DATA_DIR, NUM_WORKERS
from dataset import MnistDataModule
from mlp import NN
from helpers import plot_decision_boundary, plot_voronoi_diagram

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    # Initialize Logger
    logger = TensorBoardLogger('lightning_logs', name='MNIST')

    # Initialize Network
    model = NN(2, NUM_CLASSES, HIDDEN_LAYERS, LEARNING_RATE).to(device)

    # DataLoader setup
    dm = MnistDataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        extraction_method='wbrh'
    )

    # Initialize trainer
    trainer = pl.Trainer(
        min_epochs=1,
        max_epochs=EPOCHS,
        accelerator='cpu',
        logger=logger
    )
    trainer.fit(model, dm)
    trainer.test(model, dm)

    val_loader = dm.train_dataloader()
    data_iter = iter(val_loader)
    images, labels = next(data_iter)
    X = images.view(-1, 2).cpu().numpy()  # Assuming 2D features
    y = labels.cpu().numpy()

    # Plot decision boundary (optional visualization)
    plot_decision_boundary(model, X, y, title="Decision Boundary on MNIST (Training)", device=device)

    # Make predictions
    model.eval()
    with torch.no_grad():
        y_logits = model(images.to(device))
        y_pred = y_logits.argmax(dim=1).cpu().numpy()

    # Optionally plot Voronoi diagram if needed
    plot_voronoi_diagram(X, y_pred, n_clusters=NUM_CLASSES, y_true=y, title="Traning")

    test_loader = dm.test_dataloader()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    X = images.view(-1, 2).cpu().numpy()  # Assuming 2D features
    y = labels.cpu().numpy()

    # Plot decision boundary (optional visualization)
    plot_decision_boundary(model, X, y, title="Decision Boundary on MNIST (Test)", device=device)


# Call the main function to execute
if __name__ == "__main__":
    main()

