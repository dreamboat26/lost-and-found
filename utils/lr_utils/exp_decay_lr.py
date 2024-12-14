##
# @author Meet Patel <>
# @file Description
# @desc Created on 2023-07-09 10:02:55 pm
# @copyright MIT License
#

import sys

import numpy as np

from utils.logger_utils import logger
from utils.lr_utils.lr_scheduler import LearningRateScheduler


class ExpDecay(LearningRateScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        required_keys = [
            "burn_in_steps",
            "lr_exp_decay",
            "init_lr",
            "steps_per_epoch",
        ]

        if set(kwargs.keys()) != set(required_keys):
            missing_keys = set(required_keys).difference(set(kwargs.keys()))
            logger.debug(
                "Following keys are required for "
                f"initialization: {missing_keys}"
            )
            sys.exit()

        self.burn_in_steps = kwargs["burn_in_steps"]
        self.init_lr = kwargs["init_lr"]
        self.lr_exp_decay = kwargs["lr_exp_decay"]
        self.steps_per_epoch = kwargs["steps_per_epoch"]

    def get_lr(self, g_step: int) -> float:
        """Function to get the learning rate value based on iteration count.

        Args:
            g_step (int): Iteration count.

        Returns:
            float: Learning rate value.
        """
        if g_step < self.burn_in_steps:
            lr = (self.init_lr) * (
                g_step / self.burn_in_steps
            )  # Linear Scaling
            return lr
        else:
            return self.init_lr * np.exp(
                -(1 - self.lr_exp_decay)
                * (g_step - self.burn_in_steps)
                / self.steps_per_epoch
            )
