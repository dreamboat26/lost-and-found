##
# @author Meet Patel <>
# @file Description
# @desc Created on 2023-07-09 9:55:09 pm
# @copyright MIT License
#
import os
from abc import ABC, abstractmethod

from torch.optim import Optimizer
from tqdm import tqdm

# from misc.summary_writer import SummaryHelper


class LearningRateScheduler(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_lr(self, g_step: int) -> float:
        """Function to get the learning rate value based on iteration count.

        Args:
            g_step (int): Iteration count.

        Returns:
            float: Learning rate value.
        """
        pass

    def step(self, g_step: int, opt: Optimizer = None) -> float:
        """Function to get the learning rate and set the same to given optimizer.

        Args:
            g_step (int): Iteration count.
            opt (Optimizer, Optional): Optimizer instance. Defaults to None.

        Returns:
            float: Learning rate value.
        """
        lr = self.get_lr(g_step)

        if opt:
            for grp in opt.param_groups:
                grp["lr"] = lr

        return lr

    def plot_lr(self, op_dir_path: str, eps: int, steps_per_eps: int) -> None:
        """Function to plot the graph of the learning rate throughout the training.

        Args:
            op_dir_path (str): Directory in which Tensorboard file is to be
                saved for lr graph.
            eps (int): Epoch count.
            steps_per_eps (int): Total steps per epoch.
        """
        os.makedirs(op_dir_path, exists_ok=True)
        op_path = os.path.join(op_dir_path, "lr")
        lr_sum_writer = SummaryHelper(op_path)

        for e in tqdm(range(eps)):
            for s in range(steps_per_eps):
                if (s + 1) % 10 == 0:
                    g_step = steps_per_eps * e + s
                    lr = self.get_lr(g_step)
                    lr_sum_writer.add_summary({"lr": lr}, g_step)

        lr_sum_writer.close()
