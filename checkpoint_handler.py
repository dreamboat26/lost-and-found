
import os
from typing import Dict

import torch


class CheckpointHandler:
    def __init__(
        self, ckpt_dir: str, model_name: str = "model", max_to_keep: int = 3
    ):
        """Initializer for CheckpointHandler.
        This will save model whenever called and it will only keep track of
        last n number of epochs data only.

        Args:
            ckpt_dir (str): Directory where all the checkpoints are saved.
            model_name (str, optional): Model name to use while saving the
                checkpoint. Defaults to "model".
            max_to_keep (int, optional): Number of last checkpoints to retain.
                Defaults to 3.
        """
        self.ckpt_dir = ckpt_dir
        self.model_name = model_name
        self.max_to_keep = max_to_keep
        self.ckpt_path_history = []

    def __get_ckpt_path(self, eps: int, gen_loss: float, disc_loss: float, mean_psnr: float, mean_ssim: float) -> str:
        """Function to get the checkpoint path based on given epoch and loss value.

        Args:
            eps (int): Epoch number.
            gen_loss (float): Generator Loss value.
            disc_loss (float): Discriminator Loss value.
            mean_psnr (float): Mean PSNR value.
            mean_ssim (float): Mean SSIM value.

        Returns:
            str: Checkpoint path.
        """
        exp_id = f"epoch_{eps}_gl_{gen_loss:.4f}_dl_{disc_loss:.4f}_psnr_{mean_psnr:.4f}_ssim_{mean_ssim:.4f}" 
        ckpt_name = f"{self.model_name}_{exp_id}.pth"
        cur_ckpt_path = os.path.join(self.ckpt_dir, ckpt_name)
        return cur_ckpt_path

    def save(self, checkpoint_state: Dict) -> None:
        """Function to save the current checkpoint based on provided checkpoint
        dict.

        Args:
            checkpoint_state (Dict): Checkpoint dict which contains epoch_num,
                test_loss value, checkpoint statedict.
        """
        eps = checkpoint_state["epoch"]
        gen_loss = checkpoint_state["gen_loss"]
        disc_loss = checkpoint_state["disc_loss"]
        mean_psnr = checkpoint_state["mean_psnr"]
        mean_ssim = checkpoint_state["mean_ssim"]

        cur_ckpt_path = self.__get_ckpt_path(eps, gen_loss, disc_loss, mean_psnr, mean_ssim)

        torch.save(checkpoint_state, cur_ckpt_path)

        self.ckpt_path_history.append(cur_ckpt_path)

        if len(self.ckpt_path_history) > self.max_to_keep:
            remove_ckpt_path = self.ckpt_path_history.pop(0)
            os.remove(remove_ckpt_path)
