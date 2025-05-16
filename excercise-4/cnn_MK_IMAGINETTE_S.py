import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from helpers import plot_confusion_matrix

class Classifier(nn.Module):
    def __init__(self, input_dim=2, units=64, num_classes=10):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, units)
        self.fc2 = nn.Linear(units, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN(pl.LightningModule):
    def __init__(self, input_channels, num_classes, learning_rate=1e-3):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)

        self.conv_reduce = nn.Conv2d(16 + 32 + 64, 2, kernel_size=1, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classifier head (maps 2D features to class logits)
        self.classifier = Classifier(2, 512, num_classes)

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes)

        self.lr = learning_rate

    def forward_encoder(self, x):
        r = []
        x = F.relu(self.conv1(x))
        r.append(F.adaptive_avg_pool2d(x, (7, 7)))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        r.append(F.adaptive_avg_pool2d(x, (7, 7)))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        r.append(F.adaptive_avg_pool2d(x, (7, 7)))

        # stack outputs, reduce, flatten
        x = torch.cat(r, dim=1)
        x = self.conv_reduce(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        features_2d = x.view(x.size(0), -1)

        return features_2d

    def forward(self, x):
        features_2d = self.forward_encoder(x)

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

