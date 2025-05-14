import torch
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from config import LEARNING_RATE, INPUT_SIZE, NUM_CLASSES, HIDDEN_LAYERS, BATCH_SIZE, EPOCHS, DATA_DIR, NUM_WORKERS
from dataset import MnistDataModule, ImageNetteDataModule
from cnn_JJ_1_MNIST import CNN as CNN1_JJ
from cnn_JJ_2_MNIST import SimpleCNN as CNN2_JJ
from cnn_JJ_2_IMAGINETTE import SimpleCNN as CNN4_JJ
from cnn_JJ_1_IMAGINETTE import CNN as CNN3_JJ
from helpers import plot_decision_boundary
import torch.nn.functional as F

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    # Initialize Logger
    logger = TensorBoardLogger('lightning_logs', name='MNIST')

    # Initialize Network
    model = CNN3_JJ(3, NUM_CLASSES, LEARNING_RATE).to(device)

    # DataLoader setup
    dm = MnistDataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    im = ImageNetteDataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # Initialize trainer
    trainer = pl.Trainer(
        min_epochs=1,
        max_epochs=EPOCHS,
        accelerator='cpu',
        logger=logger
    )
    trainer.fit(model, im)
    trainer.test(model, im)

    # Plotting Decision Boundary
    # model.eval()
    # with torch.no_grad():
    #     # Get a batch of data
    #     val_loader = im.train_dataloader()
    #     images, labels = next(iter(val_loader))
    #     images = images.to(device)
    #
    #     # Forward pass: Get 2D features from the embedding layer
    #     # Apply the convolution and pooling layers
    #     x = F.relu(model.conv1(images))  # First convolution
    #     x = model.pool(x)  # First pooling layer
    #     x = F.relu(model.conv2(x))  # Second convolution
    #     x = model.pool(x)  # Second pooling layer
    #     x = F.relu(model.conv3(x))  # Third convolution (if it exists)
    #     x = model.pool(x)  # Third pooling layer
    #
    #     x = x.view(x.size(0), -1)
    #     features_2d = model.embedding(x).cpu().numpy()
    #
    #     # Convert the labels to numpy for plotting
    #     y = labels.cpu().numpy()
    #
    #     # Now plot decision boundary on 2D feature space
    #     plot_decision_boundary(model.classifier, features_2d, y, title="Decision Boundary Train", device=device)
    #
    # with torch.no_grad():
    #     # Get a batch of data
    #     val_loader = im.test_dataloader()
    #     images, labels = next(iter(val_loader))
    #     images = images.to(device)
    #
    #     # Forward pass: get 2D features from the embedding layer
    #     x = F.relu(model.conv1(images))  # First convolution
    #     x = model.pool(x)  # First pooling layer
    #     x = F.relu(model.conv2(x))  # Second convolution
    #     x = model.pool(x)  # Second pooling layer
    #     x = F.relu(model.conv3(x))  # Third convolution (if it exists)
    #     x = model.pool(x)  # Third pooling layer
    #
    #     x = x.view(x.size(0), -1)
    #     features_2d = model.embedding(x).cpu().numpy()
    #
    #     y = labels.cpu().numpy()
    #
    #     # Now plot decision boundary on 2D feature space
    #     plot_decision_boundary(model.classifier, features_2d, y, title="Decision Boundary Test", device=device)

# Call the main function to execute
if __name__ == "__main__":
    main()

