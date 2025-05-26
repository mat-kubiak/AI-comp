import numpy as np

import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

from config import LEARNING_RATE, INPUT_SIZE, NUM_CLASSES, HIDDEN_LAYERS, BATCH_SIZE, EPOCHS, DATA_DIR, NUM_WORKERS
from dataset import MnistDataModule, ImageNetteDataModule
from helpers import plot_decision_boundary

from cnn_JJ_2_MNIST import SimpleCNN as CNN_MNIST_SIMPLE
from cnn_JJ_1_MNIST import CNN as CNN_MNIST_REGULAR

from cnn_JJ_2_IMAGINETTE import SimpleCNN as CNN_IMAG_SIMPLE
from cnn_JJ_1_IMAGINETTE import CNN as CNN_IMAG_REGULAR

accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(accelerator)

if accelerator == 'cuda': # use tensor cores
    torch.set_float32_matmul_precision('medium')

def prepare_dataset(dataset, transform, sample_limit):
    kwargs = {
        'data_dir': DATA_DIR,
        'batch_size': BATCH_SIZE,
        'num_workers': NUM_WORKERS,
        'transform': transform,
        'limit_samples': sample_limit,
    }
    if dataset == 'mnist':
        return MnistDataModule(**kwargs)
    elif dataset == 'imagenette':
        return ImageNetteDataModule(**kwargs)

def prepare_model(dataset, mode):
    if dataset == 'mnist':
        if mode == 'simple':
            model = CNN_MNIST_SIMPLE(1, NUM_CLASSES, LEARNING_RATE).to(device)
        elif mode == 'regular':
            model = CNN_MNIST_REGULAR(1, NUM_CLASSES, LEARNING_RATE).to(device)

    elif dataset == 'imagenette':
        if mode == 'simple':
            model = CNN_IMAG_SIMPLE(3, NUM_CLASSES, LEARNING_RATE).to(device)
        elif mode == 'regular':
            model = CNN_IMAG_REGULAR(3, NUM_CLASSES, LEARNING_RATE).to(device)

    return model

def main():

    # how many experiments (rows) to skip, useful for resuming
    skip_n = 0

    epochs = 10
    tries = 10
    dataset = 'imagenette'
    transforms = ['default', 'JJ_color_jitter', 'MK_random_crop']
    modes = ['regular', 'simple']
    limits = [100, 200, 1000, 0]

    if skip_n == 0:
        with open('output.csv', 'a') as file:
            file.write("dataset,transform,limit,mode,acc_mean,acc_best,acc_std,accuracies\n")

    i = -1
    for transform in transforms:
            for limit in limits:

                dm = prepare_dataset(dataset, transform, limit)

                for mode in modes:
                    i = i+1
                    if i < skip_n:
                        print(f'\n### SKIPPING {dataset} {transform} {limit} {mode} ###\n')
                        continue

                    accuracies = []
                    for i_try in range(tries):

                        print(f'\n### {dataset} {transform} {limit} {mode} TRY ({i_try+1}/{tries}) ###\n')

                        model = prepare_model(dataset, mode)

                        checkpoint_callback = ModelCheckpoint(
                            monitor="val_acc",
                            mode="max",
                            save_top_k=1,
                            save_last=False,
                            filename="best-{epoch}-{val_acc:.4f}"
                        )

                        trainer = pl.Trainer(
                            min_epochs=1,
                            max_epochs=epochs,
                            accelerator=accelerator,
                            callbacks=[checkpoint_callback]
                        )

                        trainer.fit(model, dm)
                        
                        best_model_path = checkpoint_callback.best_model_path
                        model.load_state_dict(torch.load(best_model_path)["state_dict"])
                        
                        stats = trainer.test(model, dm)[0]
                        acc = stats['test_acc']
                        accuracies.append(acc)

                    # log results
                    with open('output.csv', 'a') as file:
                        accuracies_text = ','.join(str(x) for x in accuracies)

                        accuracies = np.array(accuracies)
                        acc_mean = np.mean(accuracies)
                        acc_best = np.max(accuracies)
                        acc_std = np.std(accuracies)

                        file.write(f"{dataset},{transform},{limit},{mode},{acc_mean},{acc_best},{acc_std},{accuracies_text}\n")

if __name__ == "__main__":
    main()
