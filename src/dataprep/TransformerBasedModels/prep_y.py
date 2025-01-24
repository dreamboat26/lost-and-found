from pyprojroot import here
import os
import pandas as pd
import numpy as np
import configparser
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils.utils import normalize_df
import logging

logger = logging.getLogger(__name__)
# Function does the same as the __getitem__ function in FedFormer Rrepository
# excep for the whole dataset at once


def get_data(length, data_x, data_y, data_stamp, seq_len, label_len, pred_len):
    for i in range(length):
        s_begin = i
        s_end = s_begin + seq_len
        r_begin = s_end - label_len
        r_end = r_begin + label_len + pred_len

        seq_x = data_x[s_begin:s_end]
        seq_y = data_y[r_begin:r_end]
        seq_x_mark = data_stamp[s_begin:s_end]
        seq_y_mark = data_stamp[r_begin:r_end]
        if i == 0:
            seq_x_full = seq_x[np.newaxis, :]
            seq_y_full = seq_y[np.newaxis, :]
            seq_x_mark_full = seq_x_mark[np.newaxis, :]
            seq_y_mark_full = seq_y_mark[np.newaxis, :]
        else:
            seq_x_full = np.concatenate(
                [seq_x_full, seq_x[np.newaxis, :]], axis=0)
            seq_y_full = np.concatenate(
                [seq_y_full, seq_y[np.newaxis, :]], axis=0)
            seq_x_mark_full = np.concatenate(
                [seq_x_mark_full, seq_x_mark[np.newaxis, :]], axis=0)
            seq_y_mark_full = np.concatenate(
                [seq_y_mark_full, seq_y_mark[np.newaxis, :]], axis=0)

    return seq_x_full, seq_y_full, seq_x_mark_full, seq_y_mark_full


def prepare_transformerbased_dataset():
    config = configparser.ConfigParser()
    config.read(os.path.join(here("config/dataprep.cfg")))
    config_header = "TransformerBased"
    WEATHER_DIR = here(config[config_header]["WEATHER_DIR"])  # str
    SAVING_DIR = here(config[config_header]["SAVING_DIR"])  # str
    TARGET = config[config_header]["TARGET"]  # str
    WINDOWSIZE = int(config[config_header]["WINDOWSIZE"])  # int
    LABEL_LEN = int(config[config_header]["LABEL_LEN"])  # int
    HORIZON = int(config[config_header]["HORIZON"])  # int
    TIME_ENC = int(config[config_header]["TIME_ENC"])  # int
    TEST_SIZE = float(config[config_header]["TEST_SIZE"])  # 20%
    VALID_SIZE = float(config[config_header]["VALID_SIZE"])  # 20%

    # read master table and drop wind power column
    df = pd.read_parquet(WEATHER_DIR)
    df = normalize_df(df)  # Normalize the dataframe
    df = df.reset_index()
    # separate target to be able to perform MV input, Univariate output
    df_y = df[[TARGET]]

    cols_data = df.columns[1:]  # remove date column
    df_data = df[cols_data]  # take all the columns axcept date
    data_x = df_data.values  # convert to numpy
    data_y = df_y.values
    print("data_x shape:", data_x.shape)
    print("data_y shape:", data_x.shape)

    df_stamp = df[['date']]
    df['date'] = pd.to_datetime(df['date'])
    print("df_stamp shape", df_stamp.shape)
    if TIME_ENC == 0:
        # First approach: compute 4 time related features and return a numpy array
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(
            lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        data_stamp = df_stamp.drop(columns=['date'])
        data_stamp = data_stamp.values  # convert to numpy
        print("data_stamp shape:", data_stamp.shape)
    else:
        from src.utils.timefeatures import time_features
        freq = 'h'
        data_stamp = time_features(pd.to_datetime(
            df_stamp['date'].values), freq=freq)
        print("data_stamp shape:", data_stamp.shape)
        data_stamp = data_stamp.transpose(1, 0)  # switch axis
        print("data_stamp shape:", data_stamp.shape)

    print("Dataset is ready to be converted into the sequential format:")

    print('data_x shape', data_x.shape, 'data_y shape',
          data_y.shape, 'data_stamp shape', data_stamp.shape)

    length = len(data_x) - WINDOWSIZE - HORIZON + 1
    seq_x, seq_y, seq_x_mark, seq_y_mark = get_data(
        length=length, data_x=data_x, data_y=data_y, data_stamp=data_stamp,
        seq_len=WINDOWSIZE, label_len=LABEL_LEN, pred_len=HORIZON)

    del (data_x, data_y, data_stamp)

    print('seq_x shape', seq_x.shape, 'seq_y shape', seq_y.shape,
          'seq_x_mark shape', seq_x_mark.shape, 'seq_y_mark shape', seq_y_mark.shape)

    # prepare train, test, and validation for all datasets
    print("Prepare x ...")
    x_train, x_test = train_test_split(
        seq_x, test_size=TEST_SIZE, shuffle=False)
    x_train, x_valid = train_test_split(
        x_train, test_size=VALID_SIZE, shuffle=False)

    np.save(SAVING_DIR + "/x_train", x_train)
    np.save(SAVING_DIR + "/x_test", x_test)
    np.save(SAVING_DIR + "/x_valid", x_valid)

    del (seq_x, x_train, x_test, x_valid)

    print("Prepare y ...")
    y_train, y_test = train_test_split(
        seq_y, test_size=TEST_SIZE, shuffle=False)
    # Split the train data further into train (80%) and validation (20%) sets, without shuffling.
    y_train, y_valid = train_test_split(
        y_train, test_size=VALID_SIZE, shuffle=False)

    np.save(SAVING_DIR + "/y_train", y_train)
    np.save(SAVING_DIR + "/y_test", y_test)
    np.save(SAVING_DIR + "/y_valid", y_valid)
    print("Train data shape:", "x_train:", x_train.shape, "y_train:", y_train.shape,
          "time_x_train:", time_x_train.shape, "time_y_train:", time_y_train.shape)

    del (seq_y, y_train, y_test, y_valid)

    print("Prepare time_x ...")
    time_x_train, time_x_test = train_test_split(
        seq_x_mark, test_size=TEST_SIZE, shuffle=False)
    time_x_train, time_x_valid = train_test_split(
        x_train, test_size=VALID_SIZE, shuffle=False)
    print("Test data shape:", "x_test:", x_test.shape, "y_test:", y_test.shape,
          "time_x_test:", time_x_test.shape, "time_y_test:", time_y_test.shape)

    np.save(SAVING_DIR + "/time_x_train", time_x_train)
    np.save(SAVING_DIR + "/time_x_test", time_x_test)
    np.save(SAVING_DIR + "/time_x_valid", time_x_valid)

    del (seq_x_mark, time_x_train, time_x_test, time_x_valid)

    print("Prepare time_y ...")
    time_y_train, time_y_test = train_test_split(
        seq_y_mark, test_size=TEST_SIZE, shuffle=False)
    time_y_train, time_y_valid = train_test_split(
        x_train, test_size=VALID_SIZE, shuffle=False)

    np.save(SAVING_DIR + "/time_y_train", time_y_train)
    np.save(SAVING_DIR + "/time_y_test", time_y_test)
    np.save(SAVING_DIR + "/time_y_valid", time_y_valid)

    del (seq_y_mark, time_y_train, time_y_test, time_y_valid)

    print("Valid data shape:", "x_valid:", x_valid.shape, "y_valid:", y_valid.shape,
          "time_x_valid:", time_x_valid.shape, "time_y_valid:", time_y_valid.shape)

    return logger.info(
        f"Successful Execution! Target training data is saved in {SAVING_DIR}."
    )
