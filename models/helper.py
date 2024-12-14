"""
 # @ Author: Meet Patel
 # @ Create Time: 2024-07-07 10:37:50
 # @ Modified by: Meet Patel
 # @ Modified time: 2024-07-12 22:06:15
 # @ Description:
 """

from __future__ import annotations

import os
from abc import ABC
from datetime import datetime

from tabulate import tabulate

from utils.logger_utils import logger


class Config(ABC):
    def __init__(self, init_exp=False):
        if init_exp:
            exp_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
            self.base_exp_path = f"./exp/{exp_time}"
            os.makedirs(self.base_exp_path, exist_ok=True)
            self.log_file = os.path.join(self.base_exp_path, "log.txt")

    @classmethod
    def print_config(cls: Config):
        """
        Print the static variables (class variables) of the class.
        """
        class_vars = {}
        for name, value in cls.__dict__.items():
            if callable(value):
                continue
            if isinstance(value, classmethod) or isinstance(
                value, staticmethod
            ):
                continue
            if name.startswith("__") or name == "_abc_impl":
                continue
            class_vars[name] = value

        logger.info("#" * 50)
        logger.info(f"#{cls.__name__.center(47)}#")
        data = ([name, value] for (name, value) in class_vars.items())
        table_data = tabulate(data, tablefmt="grid")
        table_data_lines = table_data.split("\n")
        for line in table_data_lines:
            logger.info(line)
        logger.info("#" * 50)

    @classmethod
    def to_dict(cls: Config):
        """
        Print the static variables (class variables) of the class.
        """
        class_vars = {}
        for name, value in cls.__dict__.items():
            if callable(value):
                continue
            if isinstance(value, classmethod) or isinstance(
                value, staticmethod
            ):
                continue
            if name.startswith("__") or name == "_abc_impl":
                continue
            class_vars[name] = value

        return class_vars


def model_factory(model_type: str, config: Config):
    """
    Get the model based on model_type and model config.

    Args:
        model_type (str): Name of the model type.
        config (Config): Model architectural config.

    Raises:
        NotImplementedError: If unsupported model_type is provided then this
            exception will be thrown.

    Returns:
        Model: Torch model as per the model_type and config.
    """
    if model_type == "gpt":
        from models.gpt import GPTModel

        return GPTModel(config)
    else:
        raise NotImplementedError(
            f"No model implemented for model type: {model_type}"
        )


def model_config_factory(model_type: str) -> Config:
    """
    Get the model config based on the model_type.

    Args:
        model_type (str): Name of the model type.

    Raises:
        NotImplementedError: If unsupported model_type is provided then this
            exception will be thrown.

    Returns:
        Config: Config class return for given model_type.
    """
    if model_type == "gpt":
        from models.gpt_model_config import GPTConfig

        return GPTConfig
    else:
        raise NotImplementedError(
            f"No model config implemented for model type: {model_type}"
        )


def train_config_factory(model_type: str) -> Config:
    """
    Get the model training config based on the model_type.

    Args:
        model_type (str): Name of the model type.

    Raises:
        NotImplementedError: If unsupported model_type is provided then this
            exception will be thrown.

    Returns:
        Config: Config class return for given model_type.
    """
    if model_type == "gpt":
        from models.gpt_train_config import GPTTrainConfig

        return GPTTrainConfig()
    else:
        raise NotImplementedError(
            f"No model config implemented for model type: {model_type}"
        )
