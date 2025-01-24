import random
import configparser
import numpy as np
import torch
from pyprojroot import here
import json
from torch.utils.data import Dataset, DataLoader
import time
import os
from src.utils.EarlyStopping import EarlyStopper
from src.models.LSTMBased.Benchmarks import VanillaLSTM, LSTMDENSE, LSTMAutoEncoder
from tqdm import tqdm
from typing import Tuple
random.seed(777)

# Set the device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the model
model = LSTMAutoEncoder().to(DEVICE)
model = torch.nn.DataParallel(model)

# print the model structure
print("Model architecture:")
print(model)
# count the number of parameters
num_params = sum(p.numel() for p in model.parameters())
print("Number of parameters in the model:", num_params)

# read the configs
config = configparser.ConfigParser()
config.read(os.path.join(here("config/training.cfg")))
DATA_DIR = here(config["TRAINING"]["DATA_DIR"])
WINDOWSIZE = int(config["TRAINING"]["WINDOWSIZE"])
HORIZON = int(config["TRAINING"]["HORIZON"])
SKIP = int(config["TRAINING"]["SKIP"])
NUM_WORKERS = int(config["TRAINING"]["NUM_WORKERS"])

# Define constants

EPOCHS = int(config["VanillaLSTM"]["EPOCHS"])
BATCH_SIZE = int(config["VanillaLSTM"]["BATCH_SIZE"])
LEARNING_RATE = float(config["VanillaLSTM"]["LEARNING_RATE"])
REDUCE_LR_PATIENCE = int(config["VanillaLSTM"]["REDUCE_LR_PATIENCE"])
REDUCE_LR_FACTOR = float(config["VanillaLSTM"]["REDUCE_LR_FACTOR"])
MIN_LR = float(config["VanillaLSTM"]["MIN_LR"])
EARLY_STOPPING_PATIENCE = int(config["VanillaLSTM"]["EARLY_STOPPING_PATIENCE"])

print("\n", "WINDOW_SIZE:", WINDOWSIZE, "|", "HORIZON:", HORIZON, "|", "SKIP:", SKIP, "|", "\n",
      "EPOCHS:", EPOCHS, "|", "BATCH_SIZE:", BATCH_SIZE, "|", "LEARNING_RATE:", LEARNING_RATE, "|", "\n",
      "REDUCE_LR_PATIENCE:", REDUCE_LR_PATIENCE, "|", "REDUCE_LR_FACTOR:", REDUCE_LR_FACTOR, "|", "\n",
      "MIN_LR:", MIN_LR, "|", "EARLY_STOPPING_PATIENCE:", EARLY_STOPPING_PATIENCE, "\n"
      )

sub_name = str(WINDOWSIZE) + "_" + str(HORIZON) + "_" + str(SKIP)

DATA_DIR = os.path.join(here(), DATA_DIR, sub_name)

x_train_path = here(f"{DATA_DIR}/x_train.npy")  # WS sequential data
y_train_path = here(f"{DATA_DIR}/y_train.npy")  # WS labels
x_feat_train = here(f"{DATA_DIR}/x_feat_train.npy")
x_train_fut_time = here(f"{DATA_DIR}/x_train_fut_time.npy")
x_train_hist_time = here(f"{DATA_DIR}/x_train_hist_time.npy")

x_valid_path = here(f"{DATA_DIR}/x_valid.npy")  # WS sequential data
y_valid_path = here(f"{DATA_DIR}/y_valid.npy")  # WS labels
x_feat_valid = here(f"{DATA_DIR}/x_feat_valid.npy")
x_valid_fut_time = here(f"{DATA_DIR}/x_valid_fut_time.npy")
x_valid_hist_time = here(f"{DATA_DIR}/x_valid_hist_time.npy")

x_test_path = here(f"{DATA_DIR}/x_test.npy")  # WS sequential data
y_test_path = here(f"{DATA_DIR}/y_test.npy")  # WS labels
x_feat_test = here(f"{DATA_DIR}/x_feat_test.npy")
x_test_fut_time = here(f"{DATA_DIR}/x_test_fut_time.npy")
x_test_hist_time = here(f"{DATA_DIR}/x_test_hist_time.npy")

train_length = np.load(y_train_path, allow_pickle=True).shape[0]
valid_length = np.load(y_valid_path, allow_pickle=True).shape[0]
test_length = np.load(y_test_path, allow_pickle=True).shape[0]

print("x_train length", train_length, "x_valid length",
      valid_length, "x_test length", test_length)


class NumpyDataset(Dataset):
    """
    Attention: this class reads the numpy arrays in mmap_mode to avoid loading them on the memory.
    """

    def __init__(self, x_path, feat_path, hist_time_path, fut_time_path, y_path, data_length):
        self.x_path = x_path
        self.feat_path = feat_path
        self.hist_time_path = hist_time_path
        self.fut_time_path = fut_time_path
        self.y_path = y_path
        self.data_length = data_length

        self.x = np.load(self.x_path, mmap_mode='r')
        self.x_Feat = np.load(self.feat_path, mmap_mode='r')
        self.x_hist_time = np.load(self.hist_time_path, mmap_mode='r')
        self.y = np.load(self.y_path, mmap_mode='r')

    def __len__(self) -> int:
        return self.data_length

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        x = np.copy(self.x[idx])
        x_Feat = np.copy(self.x_Feat[idx])
        x_hist_time = np.copy(self.x_hist_time[idx])
        x = np.concatenate([x, x_Feat, x_hist_time], axis=-1)
        # x_fut_time = np.load(self.fut_time_path, mmap_mode='r')[idx]
        # Labels
        y = np.copy(np.load(self.y_path, mmap_mode='r')[idx])
        return torch.from_numpy(x), torch.from_numpy(y)


# Create data loaders
train_dataset = NumpyDataset(
    x_train_path, x_feat_train, x_train_hist_time, x_train_fut_time, y_train_path, train_length)
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

validation_dataset = NumpyDataset(
    x_valid_path, x_feat_valid, x_valid_hist_time, x_valid_fut_time, y_valid_path, valid_length)
validation_dataloader = DataLoader(
    validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

test_dataset = NumpyDataset(
    x_test_path, x_feat_test, x_test_hist_time, x_test_fut_time, y_test_path, test_length)
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.MSELoss()

early_stopper = EarlyStopper(
    patience=EARLY_STOPPING_PATIENCE,
    min_delta=0.00001
)
optimizer_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    mode='min',  # default
    factor=REDUCE_LR_FACTOR,
    patience=REDUCE_LR_PATIENCE,
    threshold=0.000001,
    threshold_mode='rel',  # default
    cooldown=0,  # default
    min_lr=MIN_LR,  # default
    eps=1e-08,  # default
    verbose=True  # default
)
print("Starting to train...")
test_results = []
for epoch in range(EPOCHS):
    model.train()
    epoch_start_time = time.time()
    # total_loss = 0
    train_loss = 0.0
    progress_bar = tqdm(train_dataloader, total=len(
        train_dataloader), desc=f"Epoch {epoch + 1}", ncols=100)
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_dataloader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in validation_dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
        val_loss /= len(validation_dataloader)

    if early_stopper.early_stop(val_loss):
        print("Early stopping due to no improvement.")
        break

    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

        test_loss /= len(test_dataloader)
    test_results.append(test_loss)
    print('Time: {:5.2f}s | Train MSE: {:5.4f} | Val MSE: {:5.4f} | Test MSE: {:5.4f}'.format(
        (time.time() - epoch_start_time), train_loss, val_loss, test_loss))
    # scheduler.step()
    optimizer_scheduler.step(val_loss)

print("Best Test MSE:", min(test_results))
