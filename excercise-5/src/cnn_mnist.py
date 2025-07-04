import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

class CNN(pl.LightningModule):
    def __init__(self, input_channels, num_classes, learning_rate):
        super(CNN, self).__init__()
        self.save_hyperparameters()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)  # Conv Layer 1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Conv Layer 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Conv Layer 3

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 512)  # Flattened size after pooling
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")

        self.confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.lr = learning_rate

    def forward(self, x):
        # Apply convolution, pooling, and ReLU activation
        x = F.relu(self.conv1(x))  # First convolution
        x = self.pool(x)  # Max pooling
        x = F.relu(self.conv2(x))  # Second convolution
        x = self.pool(x)  # Max pooling
        x = F.relu(self.conv3(x))  # Third convolution
        x = self.pool(x)  # Max pooling

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)  # Flatten the tensor (size: batch_size x 128 * 3 * 3)

        # Fully connected layers
        x = F.relu(self.fc1(x))  # First hidden layer
        x = F.relu(self.fc2(x))  # Second hidden layer
        x = self.fc3(x)  # Output layer (logits)
        return x

    def training_step(self, batch, batch_idx):
        data, target = batch
        # Forward pass
        output = self(data)

        # Loss calculation (CrossEntropyLoss expects raw logits)
        loss = F.cross_entropy(output, target)

        # Metrics computation
        acc = self.train_acc(output, target)
        f1 = self.train_f1(output, target)
        self.confmat.update(output, target)

        self.log_dict({"train_loss": loss, "train_acc": acc, "train_f1": f1}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)

        # Loss calculation
        loss = F.cross_entropy(output, target)

        acc = self.val_acc(output, target)
        f1 = self.val_f1(output, target)
        self.confmat.update(output, target)

        self.log_dict({"val_loss": loss, "val_acc": acc, "val_f1": f1}, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)

        # Loss calculation
        loss = F.cross_entropy(output, target)
        acc = self.val_acc(output, target)
        f1 = self.val_f1(output, target)
        self.confmat.update(output, target)

        self.log_dict({"test_loss": loss, "test_acc": acc, "test_f1": f1}, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def on_train_epoch_end(self):
        if self.current_epoch == self.trainer.max_epochs - 1:
            cm = self.confmat.compute().cpu().numpy()
            self.confmat.reset()  # reset for next epoch

    def on_test_epoch_end(self):
        cm = self.confmat.compute().cpu().numpy()
        self.confmat.reset()  # reset for next epoch
