import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from helpers import plot_confusion_matrix

class CNN(pl.LightningModule):
    def __init__(self, input_channels, num_classes, learning_rate):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(32 + 64 + 128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.lr = learning_rate

    def forward(self, x):
        r = []
        x = F.relu(self.conv1(x))
        r.append(F.adaptive_avg_pool2d(x, (1, 1)))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        r.append(F.adaptive_avg_pool2d(x, (1, 1)))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        r.append(F.adaptive_avg_pool2d(x, (1, 1)))
        x = self.pool(x)

        # stack outputs and flatten
        x = torch.cat(r, dim=1)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)

        # Loss calculation (CrossEntropyLoss expects raw logits)
        loss = F.cross_entropy(output, target)

        acc = self.accuracy(output, target)
        f1 = self.f1_score(output, target)
        self.confmat.update(output, target)

        self.log_dict({"train_loss": loss, "train_acc": acc, "train_f1": f1}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)

        # Loss calculation
        loss = F.cross_entropy(output, target)

        acc = self.accuracy(output, target)
        f1 = self.f1_score(output, target)
        self.confmat.update(output, target)

        self.log_dict({"val_loss": loss, "val_acc": acc, "val_f1": f1}, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)

        # Loss calculation
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