import pandas as pd
import os
from pyprojroot import here
import configparser
import numpy as np
from src.utils.utils import extract_feature_sequences, normalize_df
from sklearn.model_selection import train_test_split
import json
import logging

logger = logging.getLogger(__name__)


def prepare_features():
    # read the configs
    config = configparser.ConfigParser()
    config.read(os.path.join(here("config/dataprep.cfg")))
    WEATHER_DIR = here(config["WEATHER"]["WEATHER_DIR"])  # str
    SAVING_DIR = here(config["WEATHER"]["SAVING_DIR"])  # str
    TARGET = config["WEATHER"]["TARGET"]  # str
    WINDOWSIZE = int(config["WEATHER"]["WINDOWSIZE"])  # int
    HORIZON = int(config["WEATHER"]["HORIZON"])  # int
    SKIP = int(config["WEATHER"]["SKIP"])  # int
    TEST_SIZE = float(config["WEATHER"]["TEST_SIZE"])  # 20%
    VALID_SIZE = float(config["WEATHER"]["VALID_SIZE"])  # 20%
    FEATURES = json.loads(config.get("WEATHER", "FEATURES"))  # List

    # read master table and drop wind power column
    df = pd.read_parquet(WEATHER_DIR)
    df = df.drop(columns=[TARGET])  # Only keep the features
    df = df[FEATURES]  # Only select the selected features
    print("Size of the dataframe:", df.shape)
    df = normalize_df(df)
    # Make sure all the columns are float32
    df = df.astype(np.float32)
    assert df.max().max() == 1 and df.min().min() == 0, "Dataframe is not normalized"

    x_feat = extract_feature_sequences(
        df=df, window_size=WINDOWSIZE, horizon=HORIZON, skip=SKIP
    )

    # Split the data into train (80%) and test (20%) sets, without shuffling.
    x_train, x_test = train_test_split(
        x_feat, test_size=TEST_SIZE, shuffle=False)
    x_train, x_valid = train_test_split(
        x_train, test_size=VALID_SIZE, shuffle=False)

    del (x_feat, df)

    sub_name = str(WINDOWSIZE) + "_" + str(HORIZON) + "_" + str(SKIP)

    print("WINDOWSIZE:", WINDOWSIZE, ", HORIZON:", HORIZON, ", SKIP:", SKIP)
    SAVING_DIR = os.path.join(here(), SAVING_DIR, sub_name)
    if not os.path.exists(SAVING_DIR):
        os.makedirs(SAVING_DIR)

    np.save(SAVING_DIR + "/x_feat_train", x_train)
    np.save(SAVING_DIR + "/x_feat_test", x_test)
    np.save(SAVING_DIR + "/x_feat_valid", x_valid)

    print("Train data shape:", "x:", x_train.shape)
    print("Test data shape:", "x:", x_test.shape)
    print("Valid data shape:", "x:", x_valid.shape)

    return logger.info(
        f"Successful Execution! Target training data is saved in {SAVING_DIR}."
    )


if __name__ == "__main__":
    prepare_features()
