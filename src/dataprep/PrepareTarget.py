from src.utils.utils import (
    normalize_df,
    generate_sequence,
)
import pandas as pd
import os
from pyprojroot import here
import configparser
from sklearn.model_selection import train_test_split
import numpy as np
import logging

logger = logging.getLogger(__name__)


def prepare_target():
    config = configparser.ConfigParser()
    config.read(here("config/dataprep.cfg"))
    WEATHER_DIR = here(config["WEATHER"]["WEATHER_DIR"])
    SAVING_DIR = here(config["WEATHER"]["SAVING_DIR"])
    TARGET = config["WEATHER"]["TARGET"]
    WINDOWSIZE = int(config["WEATHER"]["WINDOWSIZE"])
    HORIZON = int(config["WEATHER"]["HORIZON"])
    SKIP = int(config["WEATHER"]["SKIP"])
    TEST_SIZE = float(config["WEATHER"]["TEST_SIZE"])  # 20%
    VALID_SIZE = float(config["WEATHER"]["VALID_SIZE"])  # 20%

    df = pd.read_parquet(here(WEATHER_DIR), columns=[TARGET])
    # Normalize the data
    df = normalize_df(df)
    # Make sure all the columns are float32
    df = df.astype(np.float32)
    # Make sure that the data is within the correct range
    assert df.max().max() == 1 and df.min().min() == 0, "Dataframe is not normalized"

    x, y = generate_sequence(
        df=df,
        window_size=WINDOWSIZE,
        horizon=HORIZON,
        skip=SKIP,
    )
    y = y.reshape(y.shape[0], y.shape[1])
    del df

    # Split the data into train (80%) and test (20%) sets, without shuffling.
    x_train, x_test = train_test_split(x, test_size=TEST_SIZE, shuffle=False)
    # Split the train data further into train (80%) and validation (20%) sets, without shuffling.
    x_train, x_valid = train_test_split(
        x_train, test_size=VALID_SIZE, shuffle=False)
    del x

    # Split the data into train (80%) and test (20%) sets, without shuffling.
    y_train, y_test = train_test_split(y, test_size=TEST_SIZE, shuffle=False)
    # Split the train data further into train (80%) and validation (20%) sets, without shuffling.
    y_train, y_valid = train_test_split(
        y_train, test_size=VALID_SIZE, shuffle=False)
    del y

    sub_name = str(WINDOWSIZE) + "_" + str(HORIZON) + "_" + str(SKIP)

    print("WINDOWSIZE:", WINDOWSIZE, ", HORIZON:", HORIZON, ", SKIP:", SKIP)
    SAVING_DIR = os.path.join(here(), SAVING_DIR, sub_name)
    if not os.path.exists(SAVING_DIR):
        os.makedirs(SAVING_DIR)

    np.save(SAVING_DIR + "/x_train", x_train, allow_pickle=True)
    np.save(SAVING_DIR + "/x_test", x_test, allow_pickle=True)
    np.save(SAVING_DIR + "/x_valid", x_valid, allow_pickle=True)
    np.save(SAVING_DIR + "/y_train", y_train, allow_pickle=True)
    np.save(SAVING_DIR + "/y_test", y_test, allow_pickle=True)
    np.save(SAVING_DIR + "/y_valid", y_valid, allow_pickle=True)

    print("Train data shape:", "x:", x_train.shape, "y:", y_test.shape)
    print("Test data shape:", "x:", x_test.shape, "y:", y_test.shape)
    print("Valid data shape:", "x:", x_valid.shape, "y:", y_valid.shape)

    return logger.info(
        f"Successful Execution! Target training data is saved in {SAVING_DIR}."
    )


if __name__ == "__main__":
    prepare_target()
