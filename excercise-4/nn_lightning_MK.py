import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

from config import LEARNING_RATE, INPUT_SIZE, NUM_CLASSES, HIDDEN_LAYERS, BATCH_SIZE, EPOCHS, DATA_DIR, NUM_WORKERS
from dataset import MnistDataModule, ImageNetteDataModule
from helpers import plot_decision_boundary

from cnn_MK_MNIST_S import CNN as CNN_MNIST_SIMPLE
from cnn_MK_MNIST_R import CNN as CNN_MNIST_REGULAR

from cnn_MK_IMAGINETTE_S import CNN as CNN_IMAG_SIMPLE
from cnn_MK_IMAGINETTE_R import CNN as CNN_IMAG_REGULAR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from pathlib import Path
Path('models').mkdir(parents=True, exist_ok=True)

# choose the dataset & mode
DATASET = 'mnist' # ['mnist', 'imagenette']
MODE = 'regular' # ['simple', 'regular']

def prepare_mnist():
    if MODE == 'simple':
        model = CNN_MNIST_SIMPLE(1, NUM_CLASSES, LEARNING_RATE).to(device)
    elif MODE == 'regular':
        model = CNN_MNIST_REGULAR(1, NUM_CLASSES, LEARNING_RATE).to(device)

    dm = MnistDataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    return model, dm

def prepare_imagenette():
    if MODE == 'simple':
        model = CNN_IMAG_SIMPLE(3, NUM_CLASSES, LEARNING_RATE).to(device)
    elif MODE == 'regular':
        model = CNN_IMAG_REGULAR(3, NUM_CLASSES, LEARNING_RATE).to(device)

    dm = ImageNetteDataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    return model, dm

def main():
    logger = TensorBoardLogger('lightning_logs', name='MNIST')

    if DATASET == 'mnist':
        model, dm = prepare_mnist()
    elif DATASET == 'imagenette':
        model, dm = prepare_imagenette()

    trainer = pl.Trainer(
        min_epochs=1,
        max_epochs=1,
        accelerator='cpu',
        logger=logger
    )
    trainer.fit(model, dm)
    trainer.test(model, dm)

    # save model
    torch.save(model, f'models/{DATASET.capitalize()}_{MODE}.pth')

    if MODE == 'regular':
        exit()

    # Plotting Decision Boundary
    model.eval()
    with torch.no_grad():
        val_loader = dm.train_dataloader()
        images, labels = next(iter(val_loader))
        images = images.to(device)

        features_2d = model.forward_encoder(images)

        y = labels.cpu().numpy()
    
        plot_decision_boundary(model.classifier, features_2d, y, title="Decision Boundary Train", device=device)

    with torch.no_grad():
        val_loader = dm.test_dataloader()
        images, labels = next(iter(val_loader))
        images = images.to(device)
    
        features_2d = model.forward_encoder(images)

        y = labels.cpu().numpy()
    
        plot_decision_boundary(model.classifier, features_2d, y, title="Decision Boundary Test", device=device)





if __name__ == "__main__":
    main()
