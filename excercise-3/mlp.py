import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pytorch_lightning as pl
import torchmetrics
from torchmetrics.classification import MulticlassConfusionMatrix
from helpers import plot_confusion_matrix


class NN(pl.LightningModule):
    def __init__(self, input_size, num_classes, hidden_dim, learning_rate):
        super(NN, self).__init__()
        self.save_hyperparameters()

        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        self.confmat = MulticlassConfusionMatrix(num_classes=num_classes)
        self.lr = learning_rate

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    # Train Network (Pytorch Lightning)
    def training_step(self, batch, batch_idx):
        loss, scores, target = self._common_step(batch, batch_idx)
        acc = self.train_acc(scores, target)
        f1 = self.train_f1(scores, target)
        self.confmat.update(scores, target)
        self.log_dict({"loss": loss, "acc": acc, "f1": f1}, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": loss, "acc": acc, "f1": f1}

    def validation_step(self, batch, batch_idx):
        loss, scores, target = self._common_step(batch, batch_idx)
        acc = self.val_acc(scores, target)

        self.log_dict(
            {"val_loss": loss, "val_acc": acc},
            prog_bar=True,
            on_step=False,
            on_epoch=True
        )

        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        loss, scores, target = self._common_step(batch, batch_idx)
        acc = self.test_acc(scores, target)
        f1 = self.test_f1(scores, target)
        self.confmat.update(scores, target)
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