import torch
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

from config import LEARNING_RATE, INPUT_SIZE, NUM_CLASSES, HIDDEN_LAYERS, BATCH_SIZE, EPOCHS, DATA_DIR, NUM_WORKERS
from dataset import MnistDataModule
from callback import MyPrintCallback
from mlp import NN

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

if __name__ == '__main__':
    main()
