import torch
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

from config import LEARNING_RATE, INPUT_SIZE, NUM_CLASSES, HIDDEN_LAYERS, BATCH_SIZE, EPOCHS, DATA_DIR, NUM_WORKERS
from dataset import MnistDataModule
from callback import MyPrintCallback
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
        extraction_method='edges_sum'
    )

    # Initialize trainer
    trainer = pl.Trainer(
        min_epochs=1,
        max_epochs=EPOCHS,
        accelerator='cpu',
        callbacks=[MyPrintCallback()], logger=logger
    )
    trainer.fit(model, dm)
    trainer.test(model, dm)

    # This code is for 2 features only
    val_loader = dm.val_dataloader()
    data_iter = iter(val_loader)
    images, labels = next(data_iter)
    X = images.view(-1, 2).cpu().numpy()
    y = labels.cpu().numpy()
    plot_decision_boundary(model, X, y, title="Decision Boundary on MNIST (edges_sum)", device = device)

    model.eval()
    with torch.no_grad():
        y_logits = model(images.to(device))
        y_pred = y_logits.argmax(dim=1).cpu().numpy()

    plot_voronoi_diagram(X, y_pred, n_clusters=NUM_CLASSES, y_true=y)
    # End of diagrams

if __name__ == '__main__':
    main()
