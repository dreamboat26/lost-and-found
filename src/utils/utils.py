import numpy as np
import pandas as pd
from typing import List, Tuple


def feature_wise_normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.min().min()) / (df.max().max() - df.min().min())


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df_min = df.min()
    df_range = df.max() - df.min()
    return (df - df_min) / df_range


def create_sequences_from_1darray(arr: np.ndarray, window_size: int, horizon: int, skip: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(0, len(arr) - window_size - horizon + 1 - skip):
        X.append(arr[i: i + window_size])
        y.append(arr[i + window_size + skip: i + window_size + skip + horizon])
    return np.array(X), np.array(y)


def generate_sequence(df: pd.DataFrame, window_size: int, horizon: int, skip: int) -> Tuple[np.ndarray, np.ndarray]:
    result_x = []
    result_y = []
    if len(df) > horizon + skip + window_size:
        x_arr, y_arr = create_sequences_from_1darray(
            arr=df.values, window_size=window_size, horizon=horizon, skip=skip
        )
        result_x.append(x_arr)
        result_y.append(y_arr)
    concatenated_x_array = None
    for arr in result_x:
        if len(arr) != 0:
            if concatenated_x_array is None:
                concatenated_x_array = arr
            else:
                concatenated_x_array = np.concatenate(
                    [concatenated_x_array, arr], axis=0
                )

    concatenated_y_array = None
    for arr in result_y:
        if len(arr) != 0:
            if concatenated_y_array is None:
                concatenated_y_array = arr
            else:
                concatenated_y_array = np.concatenate(
                    [concatenated_y_array, arr], axis=0
                )

    return concatenated_x_array, concatenated_y_array


def extract_feature_sequences(df: pd.DataFrame, window_size: int, horizon: int, skip: int) -> np.ndarray:
    x_feats = []
    for col in df.columns:
        x_feat, _ = generate_sequence(
            df=df[[col]],
            window_size=window_size,
            horizon=horizon,
            skip=skip,
        )
        x_feats.append(x_feat)
    return np.concatenate(x_feats, axis=-1)


def add_time_related_features_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add:
    hour: 0 to 23
    weekday (monday = 0, sunday=6)
    day: 1 to 31
    month: 1 to 12
    week: 1 to 53
    year
    """
    df["hour"] = df.index.hour
    df["weekday"] = df.index.weekday
    df["day"] = df.index.day
    df["week"] = df.index.isocalendar().week
    df["month"] = df.index.month
    df["year"] = df.index.isocalendar().year
    return df


def extract_time_sequences(df: pd.DataFrame, window_size: int, horizon: int, skip: int) -> Tuple[np.ndarray, np.ndarray]:
    historical = []
    label = []
    for col in df.columns:
        historical_time, label_time = generate_sequence(
            df=df[[col]],
            window_size=window_size,
            horizon=horizon,
            skip=skip,
        )
        historical.append(historical_time)
        label.append(label_time)
        historical_arr = np.concatenate(historical, axis=-1)
        label_arr = np.concatenate(label, axis=-1)
    return historical_arr, label_arr
