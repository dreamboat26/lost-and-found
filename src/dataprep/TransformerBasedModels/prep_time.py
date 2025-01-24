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


def get_data(length, data_stamp, seq_len, label_len, pred_len):
    time_x = []
    time_y = []
    for i in range(length):
        s_begin = i
        s_end = s_begin + seq_len
        r_begin = s_end - label_len
        r_end = r_begin + label_len + pred_len

        seq_x_mark = data_stamp[s_begin:s_end]
        seq_y_mark = data_stamp[r_begin:r_end]
        seq_x_mark = seq_x_mark.reshape(
            1, seq_x_mark.shape[0], seq_x_mark.shape[1])
        seq_y_mark = seq_y_mark.reshape(
            1, seq_y_mark.shape[0], seq_y_mark.shape[1])
        time_x.append(seq_x_mark)
        time_y.append(seq_y_mark)

    time_x = np.concatenate(time_x, axis=0)
    time_y = np.concatenate(time_y, axis=0)

    return time_x, time_y


def prepare_transformerbased_time():
    config = configparser.ConfigParser()
    config.read(os.path.join(here("config/dataprep.cfg")))
    config_header = "TransformerBased"
    WEATHER_DIR = here(config[config_header]["WEATHER_DIR"])  # str
    SAVING_DIR = here(config[config_header]["SAVING_DIR"])  # str
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
    # print("data_x shape:", data_x.shape)
    # print("data_y shape:", data_x.shape)

    df_stamp = df[['date']]
    df_stamp['date'] = pd.to_datetime(df_stamp['date'])
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

    print('data_stamp shape', data_stamp.shape)

    length = len(df) - WINDOWSIZE - HORIZON + 1
    seq_x_mark, seq_y_mark = get_data(
        length=length, data_stamp=data_stamp,
        seq_len=WINDOWSIZE, label_len=LABEL_LEN, pred_len=HORIZON)

    del (data_stamp)

    print('seq_x_mark shape', seq_x_mark.shape,
          'seq_y_mark shape', seq_y_mark.shape)

    sub_name = str(WINDOWSIZE) + "_" + str(LABEL_LEN)+"_" + str(HORIZON)

    print("WINDOWSIZE:", WINDOWSIZE, ", HORIZON:",
          HORIZON, ", LABEL_LEN:", LABEL_LEN)
    SAVING_DIR = os.path.join(here(), SAVING_DIR, sub_name)
    if not os.path.exists(SAVING_DIR):
        os.makedirs(SAVING_DIR)

    print("Prepare time_x ...")
    time_x_train, time_x_test = train_test_split(
        seq_x_mark, test_size=TEST_SIZE, shuffle=False)
    time_x_train, time_x_valid = train_test_split(
        time_x_train, test_size=VALID_SIZE, shuffle=False)
    print("time_x_train:", time_x_train.shape, "time_x_test:", time_x_test.shape,
          "time_x_valid:", time_x_valid.shape)

    np.save(SAVING_DIR + "/time_x_train", time_x_train)
    np.save(SAVING_DIR + "/time_x_test", time_x_test)
    np.save(SAVING_DIR + "/time_x_valid", time_x_valid)

    del (time_x_train, time_x_test, time_x_valid)

    print("Prepare time_y ...")
    time_y_train, time_y_test = train_test_split(
        seq_y_mark, test_size=TEST_SIZE, shuffle=False)
    time_y_train, time_y_valid = train_test_split(
        time_y_train, test_size=VALID_SIZE, shuffle=False)
    print("time_y_train:", time_y_train.shape, "time_y_test:", time_y_test.shape,
          "time_y_valid:", time_y_valid.shape)

    np.save(SAVING_DIR + "/time_y_train", time_y_train)
    np.save(SAVING_DIR + "/time_y_test", time_y_test)
    np.save(SAVING_DIR + "/time_y_valid", time_y_valid)

    del (seq_y_mark, time_y_train, time_y_test, time_y_valid)

    return logger.info(
        f"Successful Execution! Target training data is saved in {SAVING_DIR}."
    )
