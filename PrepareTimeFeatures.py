from src.utils.utils import (
    normalize_df,
    add_time_related_features_to_df,
    extract_time_sequences,
)
import json
import configparser
import os
from pyprojroot import here
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def prepare_time_features():
    config = configparser.ConfigParser()
    config.read(here("config/dataprep.cfg"))
    WEATHER_DIR = here(config["WEATHER"]["WEATHER_DIR"])
    SAVING_DIR = here(config["WEATHER"]["SAVING_DIR"])
    TARGET = config["WEATHER"]["TARGET"]  # str
    WINDOWSIZE = int(config["WEATHER"]["WINDOWSIZE"])  # int
    HORIZON = int(config["WEATHER"]["HORIZON"])  # int
    SKIP = int(config["WEATHER"]["SKIP"])  # int
    TEST_SIZE = float(config["WEATHER"]["TEST_SIZE"])  # 20%
    VALID_SIZE = float(config["WEATHER"]["VALID_SIZE"])  # 20%
    TIME_FEATURES = json.loads(config.get("WEATHER", "TIME_FEATURES"))

    # read master table and drop wind power column
    df = pd.read_parquet(here(WEATHER_DIR), columns=[TARGET])
    df = add_time_related_features_to_df(df)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"].values / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"].values / 24.0)
    df = df.drop(columns=["hour"])

    df["day_sin"] = np.sin(2 * np.pi * df["day"].values / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day"].values / 31)
    df = df.drop(columns=["day"])
    df["week_sin"] = np.sin(2 * np.pi * df["week"].values / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week"].values / 52)

    df["month_sin"] = np.sin(2 * np.pi * df["month"].values / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"].values / 12)
    df = df[TIME_FEATURES]
    df = normalize_df(df)
    df = df.astype(np.float32)

    assert df.max().max() == 1 and df.min().min() == 0, "Dataframe is not normalized"

    historical_arr, label_arr = extract_time_sequences(
        df=df, window_size=WINDOWSIZE, horizon=HORIZON, skip=SKIP
    )
    del df
    historical_arr = historical_arr.astype(np.float32)
    label_arr = label_arr.astype(np.float32)

    # Split the data into train (80%) and test (20%) sets, without shuffling.
    x_train_hist_time, x_test_hist_time = train_test_split(
        historical_arr, test_size=TEST_SIZE, shuffle=False
    )
    # Split the train data further into train (80%) and validation (20%) sets, without shuffling.
    x_train_hist_time, x_valid_hist_time = train_test_split(
        x_train_hist_time, test_size=VALID_SIZE, shuffle=False
    )
    del historical_arr
    # Split the data into train (80%) and test (20%) sets, without shuffling.
    x_train_fut_time, x_test_fut_time = train_test_split(
        label_arr, test_size=TEST_SIZE, shuffle=False
    )
    # Split the train data further into train (80%) and validation (20%) sets, without shuffling.
    x_train_fut_time, x_valid_fut_time = train_test_split(
        x_train_fut_time, test_size=VALID_SIZE, shuffle=False
    )
    del label_arr

    sub_name = str(WINDOWSIZE) + "_" + str(HORIZON) + "_" + str(SKIP)

    print("WINDOWSIZE:", WINDOWSIZE, ", HORIZON:", HORIZON, ", SKIP:", SKIP)
    SAVING_DIR = os.path.join(here(), SAVING_DIR, sub_name)
    if not os.path.exists(SAVING_DIR):
        os.makedirs(SAVING_DIR)

    np.save(SAVING_DIR + "/x_train_hist_time", x_train_hist_time)
    np.save(SAVING_DIR + "/x_test_hist_time", x_test_hist_time)
    np.save(SAVING_DIR + "/x_valid_hist_time", x_valid_hist_time)
    np.save(SAVING_DIR + "/x_train_fut_time", x_train_fut_time)
    np.save(SAVING_DIR + "/x_test_fut_time", x_test_fut_time)
    np.save(SAVING_DIR + "/x_valid_fut_time", x_valid_fut_time)

    print("x_train_hist_time:", x_train_hist_time.shape)
    print("x_test_hist_time:", x_test_hist_time.shape)
    print("x_valid_hist_time:", x_valid_hist_time.shape)
    print("x_train_fut_time:", x_train_fut_time.shape)
    print("x_test_fut_time:", x_test_fut_time.shape)
    print("x_valid_fut_time:", x_valid_fut_time.shape)

    return logger.info(
        f"Successful Execution! Target training data is saved in {SAVING_DIR}."
    )


if __name__ == "__main__":
    prepare_time_features()
