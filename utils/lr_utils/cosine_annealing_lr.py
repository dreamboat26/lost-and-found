
import sys

import numpy as np

from utils.logger_utils import logger
from utils.lr_utils.lr_scheduler import LearningRateScheduler


class CosineAnnealing(LearningRateScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        required_keys = [
            "warmup_epochs",
            "epochs",
            "init_lr",
            "steps_per_epoch",
        ]

        if set(kwargs.keys()) != set(required_keys):
            missing_keys = set(required_keys).difference(set(kwargs.keys()))
            logger.error(
                "Following keys are required for "
                f"initialization: {missing_keys}"
            )
            sys.exit()

        self.warmup_epochs = kwargs["warmup_epochs"]
        self.epochs = kwargs["epochs"]
        self.init_lr = kwargs["init_lr"]
        self.steps_per_epoch = kwargs["steps_per_epoch"]
        self.burn_in_steps = self.warmup_epochs * self.steps_per_epoch
        self.cosine_iters = (
            self.steps_per_epoch * self.epochs - self.burn_in_steps
        )

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
            return (
                self.init_lr
                * 0.5
                * (
                    1
                    + np.cos(
                        np.pi
                        * (g_step - self.burn_in_steps)
                        / self.cosine_iters
                    )
                )
            )
