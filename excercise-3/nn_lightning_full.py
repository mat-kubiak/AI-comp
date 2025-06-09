import torch
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, silhouette_score, \
    pairwise_distances

from pytorch_lightning.callbacks import ModelCheckpoint
from config import LEARNING_RATE, INPUT_SIZE, NUM_CLASSES, HIDDEN_LAYERS, BATCH_SIZE, EPOCHS, DATA_DIR, NUM_WORKERS
from dataset import MnistDataModule
from mlp import NN
from helpers import plot_decision_boundary, plot_voronoi_diagram

# Set Device
accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(accelerator)

def main():

    # Initialize Logger
    logger = TensorBoardLogger('lightning_logs', name='MNIST')

    # Initialize Network
    model = NN(INPUT_SIZE, NUM_CLASSES, HIDDEN_LAYERS, LEARNING_RATE).to(device)

    ext_method = 'flatten'

    # DataLoader setup
    dm = MnistDataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        extraction_method=ext_method
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./checkpoints/",
        filename=f"mnist_{ext_method}_best",
        save_top_k=1,
        mode="min",
    )

    # Initialize trainer
    trainer = pl.Trainer(
        min_epochs=1,
        max_epochs=EPOCHS,
        accelerator=accelerator,
        callbacks=[checkpoint_callback],
        logger=logger
    )
    trainer.fit(model, dm)

    # print epoch
    checkpoint = torch.load(checkpoint_callback.best_model_path, map_location="cpu")
    print(f"Best checkpoint was saved at epoch: {checkpoint['epoch']}")

    # load best model again for testing
    model = NN.load_from_checkpoint(checkpoint_callback.best_model_path).to(device)
    trainer.test(model, dm)

# Call the main function to execute
if __name__ == "__main__":
    main()
