import os

# Hyper Parameters
INPUT_SIZE = 784
NUM_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64
HIDDEN_LAYERS = 150
EPOCHS = 60

# Data Set
NUM_WORKERS = os.cpu_count() - 1
DATA_DIR = "./dataset"
