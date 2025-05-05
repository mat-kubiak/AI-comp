import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pytorch_lightning as pl
import torchmetrics

class NN(pl.LightningModule):
    def __init__(self, input_size, num_classes, hidden_dim, learning_rate):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.lr = learning_rate

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    # Train Network (Pytorch Lightning)
    def training_step(self, batch, batch_idx):
        loss, scores, target = self._common_step(batch, batch_idx)
        acc = self.accuracy(scores, target)
        f1 = self.f1_score(scores, target)
        self.log_dict({"loss": loss, "acc": acc, "f1": f1}, prog_bar=True, on_step=False, on_epoch=True)

        return {"loss": loss, "acc": acc, "f1": f1}

    def validation_step(self, batch, batch_idx):
        loss, scores, target = self._common_step(batch, batch_idx)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, target = self._common_step(batch, batch_idx)
        acc = self.accuracy(scores, target)
        f1 = self.f1_score(scores, target)
        self.log_dict({"loss": loss, "acc": acc, "f1": f1}, prog_bar=True, on_step=False, on_epoch=True)

        return {"loss": loss, "acc": acc, "f1": f1}

    def _common_step(self, batch, batch_idx):
        data, target = batch
        data = data.reshape(data.shape[0], -1)
        scores = self.forward(data)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(scores, target)
        return loss, scores, target

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.lr)