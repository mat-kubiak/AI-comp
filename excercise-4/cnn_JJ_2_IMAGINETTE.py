import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from helpers import plot_confusion_matrix


class SimpleCNN(pl.LightningModule):
    def __init__(self, input_channels, num_classes, learning_rate=1e-3):
        super(SimpleCNN, self).__init__()

        # Convolutional layers (increased number of filters for Imagenette)
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)  # Increased filters
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Increased filters
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Added another conv layer

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # After three poolings (for 160x160 input), this will give 64 * 20 * 20 features
        self.embedding = nn.Linear(64 * 20 * 20, 2)  # Maps to 2D features (after pooling and flattening)

        # Classifier head (maps 2D features to class logits)
        self.classifier = nn.Linear(2, num_classes)

        # Metrics
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes)

        self.lr = learning_rate

    def forward(self, x):
        # Convolution + ReLU + Pooling (applied 3 times now)
        x = F.relu(self.conv1(x))  # First convolution
        x = self.pool(x)  # Max pooling
        x = F.relu(self.conv2(x))  # Second convolution
        x = self.pool(x)  # Max pooling
        x = F.relu(self.conv3(x))  # Third convolution
        x = self.pool(x)  # Max pooling

        # Flatten the tensor and pass through the embedding layer (2D features)
        x = x.view(x.size(0), -1)  # Flattening the tensor

        # Flatten the tensor and pass through the embedding layer (2D features)
        x = x.view(x.size(0), -1)  # Flattening the tensor
        features_2d = self.embedding(x)

        # Pass the 2D features to the classifier
        logits = self.classifier(features_2d)
        return logits

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = F.cross_entropy(output, target)

        acc = self.accuracy(output, target)
        f1 = self.f1_score(output, target)
        self.confmat.update(output, target)

        self.log_dict({"train_loss": loss, "train_acc": acc, "train_f1": f1}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = F.cross_entropy(output, target)

        acc = self.accuracy(output, target)
        f1 = self.f1_score(output, target)
        self.confmat.update(output, target)

        self.log_dict({"val_loss": loss, "val_acc": acc, "val_f1": f1}, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = F.cross_entropy(output, target)

        acc = self.accuracy(output, target)
        f1 = self.f1_score(output, target)
        self.confmat.update(output, target)

        self.log_dict({"test_loss": loss, "test_acc": acc, "test_f1": f1}, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def on_train_epoch_end(self):
        if self.current_epoch == self.trainer.max_epochs - 1:
            cm = self.confmat.compute().cpu().numpy()
            self.confmat.reset()  # reset for next epoch

            class_names = [str(i) for i in range(10)]
            plot_confusion_matrix(cm, class_names, title="Confusion Matrix Train")

    def on_test_epoch_end(self):
        cm = self.confmat.compute().cpu().numpy()
        self.confmat.reset()  # reset for next epoch

        class_names = [str(i) for i in range(10)]
        plot_confusion_matrix(cm, class_names, title="Confusion Matrix Test")

