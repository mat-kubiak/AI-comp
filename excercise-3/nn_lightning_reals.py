import torch
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, silhouette_score, \
    pairwise_distances

from pytorch_lightning.callbacks import ModelCheckpoint
from config import LEARNING_RATE, INPUT_SIZE, NUM_CLASSES, HIDDEN_LAYERS, BATCH_SIZE, EPOCHS, DATA_DIR, NUM_WORKERS
from dataset import MnistDataModule, SklearnDataModule
from mlp import NN
from helpers import plot_decision_boundary, plot_voronoi_diagram

accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(accelerator)

def choose_dataset():
    print("Choose a dataset:")
    print("1 - Iris")
    print("2 - Wine")
    print("3 - Breast Cancer Wisconsin")
    choice = input("Enter number (1/2/3): ")

    if choice == '1':
        name = "Iris"
    elif choice == '2':
        name = "Wine"
    elif choice == '3':
        name = "Breast Cancer Wisconsin"
    else:
        print("Invalid choice. Defaulting to Iris.")
        name = "Iris"

    return name

def main():
    dataset_name = choose_dataset()

    # Initialize Logger
    logger = TensorBoardLogger('lightning_logs', name='MNIST')

    # DataLoader setup
    dm = SklearnDataModule(dataset_name, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    dm.prepare_data()

    # Initialize Network
    model = NN(dm.input_size, dm.num_classes, HIDDEN_LAYERS, LEARNING_RATE).to(device)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./checkpoints/",
        filename=f"{dataset_name.replace(" ", "_").lower()}_best",
        save_top_k=1,
        mode="min",
    )

    # Initialize trainer
    trainer = pl.Trainer(
        min_epochs=1,
        max_epochs=EPOCHS,
        accelerator=accelerator,
        callbacks=[checkpoint_callback],
        logger=logger
    )
    trainer.fit(model, dm)

    # load best model again for testing
    model = NN.load_from_checkpoint(checkpoint_callback.best_model_path).to(device)
    trainer.test(model, dm)

    val_loader = dm.train_dataloader()
    data_iter = iter(val_loader)
    images, labels = next(data_iter)
    y = labels.cpu().numpy()

    # Make predictions
    model.eval()
    with torch.no_grad():
        y_logits = model(images.to(device))
        y_pred = y_logits.argmax(dim=1).cpu().numpy()

    ari = adjusted_rand_score(y, y_pred)
    homogeneity = homogeneity_score(y, y_pred)
    completeness = completeness_score(y, y_pred)

    # Print the scores
    print(f"Adjusted Rand Index (ARI): {ari}")
    print(f"Homogeneity: {homogeneity}")
    print(f"Completeness: {completeness}")

# Call the main function to execute
if __name__ == "__main__":
    main()

