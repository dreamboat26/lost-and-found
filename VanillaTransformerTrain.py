# %%
import random
import configparser
import numpy as np
import torch
from pyprojroot import here
from torch.utils.data import Dataset, DataLoader
import time
import os
from src.utils.EarlyStopping import EarlyStopper
from src.models.Transformers.Transformer import Model
from tqdm import tqdm
from typing import Tuple
random.seed(777)

"""
Informer performs multi-input multi-output prediction. 
However, we are only interested in the predicted values of windspeed. 
Therefore, we will evaluate the model's performance using multi-input single-output metrics.
"""
# Set the device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on:", DEVICE)


def get_the_target(tensor):
    last_12_values = tensor[:, -12:, :]
    first_column = last_12_values[:, :, 0:1]
    return first_column.squeeze()


class Configs(object):
    config = configparser.ConfigParser()
    config.read(os.path.join(here("config/training.cfg")))
    config_header = "Transformer"
    seq_len = int(config[config_header]["seq_len"])  # windowsize or lookback
    # labeld samples attached to horizon
    label_len = int(config[config_header]["label_len"])
    pred_len = int(config[config_header]["pred_len"])  # horizon
    output_attention = config.getboolean(config_header, 'output_attention')
    enc_in = int(config[config_header]["enc_in"])  # num input features
    # should be the same as enc in
    dec_in = int(config[config_header]["dec_in"])
    d_model = int(config[config_header]["d_model"])  # 16
    embed = config[config_header]["embed"]
    dropout = float(config[config_header]["dropout"])
    freq = config[config_header]["freq"]
    factor = int(config[config_header]["factor"])
    n_heads = int(config[config_header]["n_heads"])
    d_ff = int(config[config_header]["d_ff"])  # 16
    e_layers = int(config[config_header]["e_layers"])  # 2
    d_layers = int(config[config_header]["d_layers"])  # 1
    c_out = int(config[config_header]["c_out"])  # number of output features
    # 1 means univariate prediction
    activation = config[config_header]["activation"]


model_configs = Configs()
model = Model(model_configs).to(DEVICE)
model = torch.nn.DataParallel(model)
# %%
# print the model structure
# print("Model architecture:")
# print(model)

# # Instantiate the model:
print('parameter number is {}'.format(sum(p.numel()
      for p in model.parameters())))
# enc = torch.randn([3, 168, 12])
# enc_mark = torch.randn([3, 168, 4])

# dec = torch.randn([3, 60, 12])
# dec_mark = torch.randn([3, 60, 4])
# out = model.forward(enc, enc_mark, dec, dec_mark)
# print(out.shape)
# # count the number of parameters
# num_params = sum(p.numel() for p in model.parameters())
# print("Number of parameters in the model:", num_params)

# %%
# read the configs
config = configparser.ConfigParser()
config.read(os.path.join(here("config/training.cfg")))
config_header = "Informer"
DATA_DIR = here(config[config_header]["DATA_DIR"])
NUM_WORKERS = int(config[config_header]["NUM_WORKERS"])

# Define constants
EPOCHS = int(config[config_header]["EPOCHS"])
BATCH_SIZE = int(config[config_header]["BATCH_SIZE"])
LEARNING_RATE = float(config[config_header]["LEARNING_RATE"])
REDUCE_LR_PATIENCE = int(config[config_header]["REDUCE_LR_PATIENCE"])
REDUCE_LR_FACTOR = float(config[config_header]["REDUCE_LR_FACTOR"])
MIN_LR = float(config[config_header]["MIN_LR"])
EARLY_STOPPING_PATIENCE = int(config[config_header]["EARLY_STOPPING_PATIENCE"])

print("\n", "WINDOW_SIZE:", model_configs.seq_len, "|", "HORIZON:", model_configs.pred_len, "|", "LABEL_LEN:", model_configs.label_len, "|", "\n",
      "EPOCHS:", EPOCHS, "|", "BATCH_SIZE:", BATCH_SIZE, "|", "LEARNING_RATE:", LEARNING_RATE, "|", "\n",
      "REDUCE_LR_PATIENCE:", REDUCE_LR_PATIENCE, "|", "REDUCE_LR_FACTOR:", REDUCE_LR_FACTOR, "|", "\n",
      "MIN_LR:", MIN_LR, "|", "EARLY_STOPPING_PATIENCE:", EARLY_STOPPING_PATIENCE, "\n"
      )

sub_name = str(model_configs.seq_len) + "_" + str(model_configs.label_len) + \
    "_" + str(model_configs.pred_len)
DATA_DIR = os.path.join(here(), DATA_DIR, sub_name)

x_train_path = here(f"{DATA_DIR}/x_train.npy")  # WS sequential data
y_train_path = here(f"{DATA_DIR}/y_train.npy")  # WS labels
time_x_train_path = here(f"{DATA_DIR}/time_x_train.npy")
time_y_train_path = here(f"{DATA_DIR}/time_y_train.npy")

x_valid_path = here(f"{DATA_DIR}/x_valid.npy")  # WS sequential data
y_valid_path = here(f"{DATA_DIR}/y_valid.npy")  # WS labels
time_x_valid_path = here(f"{DATA_DIR}/time_x_valid.npy")
time_y_valid_path = here(f"{DATA_DIR}/time_y_valid.npy")

x_test_path = here(f"{DATA_DIR}/x_test.npy")  # WS sequential data
y_test_path = here(f"{DATA_DIR}/y_test.npy")  # WS labels
time_x_test_path = here(f"{DATA_DIR}/time_x_test.npy")
time_y_test_path = here(f"{DATA_DIR}/time_y_test.npy")

train_length = np.load(y_train_path, allow_pickle=True).shape[0]
valid_length = np.load(y_valid_path, allow_pickle=True).shape[0]
test_length = np.load(y_test_path, allow_pickle=True).shape[0]

print("x_train length", train_length, "x_valid length",
      valid_length, "x_test length", test_length)


class NumpyDataset(Dataset):
    """
    Attention: this class reads the numpy arrays in mmap_mode to avoid loading them on the memory.
    """

    def __init__(self, x_path, time_x_path, y_path, time_y_path, data_length):
        self.x_path = x_path
        self.time_x_path = time_x_path
        self.time_y_path = time_y_path
        self.y_path = y_path
        self.data_length = data_length

        self.x = np.load(self.x_path, mmap_mode='r')
        self.time_x = np.load(self.time_x_path, mmap_mode='r')

        self.y = np.load(self.y_path, mmap_mode='r')
        self.time_y = np.load(self.time_y_path, mmap_mode='r')

    def __len__(self) -> int:
        return self.data_length

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        x = np.copy(self.x[idx])
        time_x = np.copy(self.time_x[idx])
        y = np.copy(self.y[idx])
        time_y = np.copy(self.time_y[idx])

        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(time_x), torch.from_numpy(time_y)


# Create data loaders
train_dataset = NumpyDataset(
    x_train_path, time_x_train_path, y_train_path, time_y_train_path, train_length)
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

validation_dataset = NumpyDataset(
    x_valid_path, time_x_valid_path, y_valid_path, time_y_valid_path, valid_length)
validation_dataloader = DataLoader(
    validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

test_dataset = NumpyDataset(
    x_test_path, time_x_test_path, y_test_path, time_y_test_path, test_length)
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
train_results = []
valid_results = []
best_val_loss = float(0.005)
for epoch in range(EPOCHS):
    iter_count = 0
    model.train()
    epoch_start_time = time.time()
    # total_loss = 0
    train_loss = 0.0
    progress_bar = tqdm(train_dataloader, total=len(
        train_dataloader), desc=f"Epoch {epoch + 1}", ncols=100)
    for batch_x, batch_y, batch_x_mark, batch_y_mark in progress_bar:
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch_x.to(DEVICE), batch_y.to(
            DEVICE), batch_x_mark.to(DEVICE), batch_y_mark.to(DEVICE)
        # print(batch_x.shape, batch_y.shape,
        #   batch_x_mark.shape, batch_y_mark.shape)
        optimizer.zero_grad()
        # decoder input
        dec_inp = torch.zeros_like(
            batch_y[:, -model_configs.pred_len:, :]).float()
        dec_inp = torch.cat(
            [batch_y[:, :model_configs.label_len, :], dec_inp], dim=1).float().to(DEVICE)

        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        y_pred = get_the_target(outputs)
        y_true = get_the_target(batch_y)

        # outputs = outputs[:, 0:1, :]  # Getting the first feature (windspeed)
        # batch_y = batch_y[:, 0:1, :]  # Getting the first feature (windspeed)

        # outputs = outputs.squeeze()
        # batch_y = batch_y.squeeze()

        # break
        # loss = criterion(outputs, batch_y[:, -model_configs.pred_len:])
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    # break
    train_loss /= len(train_dataloader)
    train_results.append(train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in validation_dataloader:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch_x.to(DEVICE), batch_y.to(
                DEVICE), batch_x_mark.to(DEVICE), batch_y_mark.to(DEVICE)

            # decoder input
            dec_inp = torch.zeros_like(
                batch_y[:, -model_configs.pred_len:, :]).float()
            dec_inp = torch.cat(
                [batch_y[:, :model_configs.label_len, :], dec_inp], dim=1).float().to(DEVICE)
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            y_pred = get_the_target(outputs)
            y_true = get_the_target(batch_y)

            # outputs = outputs[:, 0:1, :]  # Getting the first feature (windspeed)
            # batch_y = batch_y[:, 0:1, :]  # Getting the first feature (windspeed)

            # outputs = outputs.squeeze()
            # batch_y = batch_y.squeeze()

            # break
            # loss = criterion(outputs, batch_y[:, -model_configs.pred_len:])
            loss = criterion(y_pred, y_true)
            val_loss += loss.item()
        val_loss /= len(validation_dataloader)
        valid_results.append(val_loss)
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), 'model/vanilla_transformer.pt')
        best_val_loss = val_loss
        print("Model is saved.")
    if early_stopper.early_stop(val_loss):
        print("Early stopping due to no improvement in val_loss.")
        break

    test_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_dataloader:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch_x.to(DEVICE), batch_y.to(
                DEVICE), batch_x_mark.to(DEVICE), batch_y_mark.to(DEVICE)

            # decoder input
            dec_inp = torch.zeros_like(
                batch_y[:, -model_configs.pred_len:, :]).float()
            dec_inp = torch.cat(
                [batch_y[:, :model_configs.label_len, :], dec_inp], dim=1).float().to(DEVICE)

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            y_pred = get_the_target(outputs)
            y_true = get_the_target(batch_y)

            # outputs = outputs[:, 0:1, :]  # Getting the first feature (windspeed)
            # batch_y = batch_y[:, 0:1, :]  # Getting the first feature (windspeed)

            # outputs = outputs.squeeze()
            # batch_y = batch_y.squeeze()

            # break
            # loss = criterion(outputs, batch_y[:, -model_configs.pred_len:])
            loss = criterion(y_pred, y_true)
            # handle the last batch separately
            # if batch_y.shape[0] < BATCH_SIZE:
            #     loss = criterion(outputs,
            #                      batch_y[-model_configs.pred_len:])
            # else:
            #     loss = criterion(outputs,
            #                      batch_y[:, -model_configs.pred_len:])
            test_loss += loss.item()

        test_loss /= len(test_dataloader)
    test_results.append(test_loss)
    print('Time: {:5.2f}s | Train MSE: {:5.4f} | Val MSE: {:5.4f} | Test MSE: {:5.4f}'.format(
        (time.time() - epoch_start_time), train_loss, val_loss, test_loss))
    # scheduler.step()
    optimizer_scheduler.step(val_loss)

print("Best MSE:", "train:", min(train_results), "valid:",
      min(valid_results), "Test:", min(test_results))
