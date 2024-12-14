
import argparse

import torch
from torch import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from dataset_helper import DatasetHelper
from models.helper import (
    model_config_factory,
    model_factory,
    train_config_factory,
)
from utils.checkpoint_handler import CheckpointHandler
from utils.logger_utils import configure_logging, logger
from utils.misc import get_tokenizer, init_wandb, lr_scheduler_factory


def run(model_type):
    train_config = train_config_factory(model_type)
    exp_path = train_config.base_exp_path
    configure_logging(train_config.log_file)
    cuda = torch.device(train_config.device)

    model_config = model_config_factory(model_type)
    model_config.max_seq_len = (
        train_config.max_seq_len
    )  # For Positional Embeddings.
    model = model_factory(model_type, model_config)
    model.to(cuda)

    tokenizer = get_tokenizer(model_type)

    train_helper = DatasetHelper(
        tokenizer,
        train_config.batch_size,
        train_config.avg_seq_len_in_batch,
        train_config.max_seq_len,
        train_config.num_workers,
        train_config.persistent_workers,
        "train",
    )
    train_loader = train_helper.get_loader()
    valid_helper = DatasetHelper(
        tokenizer,
        train_config.batch_size,
        train_config.avg_seq_len_in_batch,
        train_config.max_seq_len,
        train_config.num_workers,
        train_config.persistent_workers,
        "validation",
    )
    valid_loader = valid_helper.get_loader()

    ckpt_handler = CheckpointHandler(exp_path, "model", max_to_keep=3)
    lr_scheduler = lr_scheduler_factory(
        train_config.lr_scheduler_type,
        init_lr=train_config.init_lr,
        epochs=train_config.num_epochs,
        warmup_epochs=train_config.warmup_epochs,
        steps_per_epoch=len(train_loader),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, weight_decay=0.1)
    loss_fn = CrossEntropyLoss(
        reduction="mean",
        label_smoothing=train_config.label_smoothing,
        ignore_index=tokenizer.convert_tokens_to_ids(tokenizer.pad_token),
    )

    scaler = GradScaler()

    if train_config.use_wandb:
        import wandb

        init_wandb(train_config, model_config, train_config.resume_wandb_id)
    g_step = 0
    if train_config.use_wandb and train_config.track_gradients:
        wandb.watch(model)
    for eps_num in range(train_config.num_epochs):
        model.train()
        for batch_idx, (input_ids, attn_mask, labels) in enumerate(
            train_loader
        ):
            optimizer.zero_grad()
            batch_size = input_ids.shape[0]
            input_ids = input_ids.to(cuda, non_blocking=True)
            attn_mask = attn_mask.to(cuda, non_blocking=True)
            labels = labels.to(cuda, non_blocking=True)

            with autocast(
                device_type=train_config.device,
                enabled=train_config.fp16_training,
            ):
                logits = model(input_ids, attn_mask)

                logits = logits.view(-1, logits.shape[2])
                labels = labels.view(-1).to(torch.long)

                # We would take mean across all sequence length and all batches.
                loss = loss_fn(logits, labels)
                ppl = torch.exp(loss)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            lr = lr_scheduler.step(g_step, optimizer)

            logger.info(
                f"Epoch: {eps_num+1}/{train_config.num_epochs}, Batch: {batch_idx}/{len(train_loader)}, "
                f"Batch Size: {batch_size}, Loss: {loss:.4f}, "
                f"PPL: {ppl:.4f}, LR: {lr:.4f}"
            )
            metrics = {
                "Epoch": eps_num + 1,
                "Batch": batch_idx + 1,
                "Loss": loss,
                "Perplexity": ppl,
                "LR": lr,
            }
            if train_config.use_wandb:
                wandb.log(metrics, step=g_step)
            g_step += 1

        model.eval()
        total_eval_loss = 0
        total_eval_ppl = 0
        with torch.no_grad():
            for input_ids, attn_mask, labels in tqdm(valid_loader):
                input_ids = input_ids.to(cuda, non_blocking=True)
                attn_mask = attn_mask.to(cuda, non_blocking=True)
                labels = labels.to(cuda, non_blocking=True)

                logits = model(input_ids, attn_mask)

                batch_size = logits.shape[0]
                logits = logits.view(-1, logits.shape[2])
                labels = labels.view(-1).to(torch.long)

                # We would take mean across all sequence length and all batches.
                loss = loss_fn(logits, labels)
                ppl = torch.exp(loss)
                total_eval_loss += loss.item()
                total_eval_ppl += ppl.item()

        avg_eval_loss = total_eval_loss / len(valid_loader)
        avg_eval_ppl = total_eval_ppl / len(valid_loader)
        logger.info(
            f"Epoch {eps_num+1}, Evaluation Loss: {avg_eval_loss:.4f}, "
            f"Evaluation Perplexity: {avg_eval_ppl:.4f}"
        )
        if train_config.use_wandb:
            metrics = {"Test Loss": loss}
            wandb.log(metrics, step=g_step)

        # Save the model
        torch.save(model.state_dict(), "model.pth")

        checkpoint = {
            "epoch": eps_num,
            "global_step": g_step,
            "test_loss": avg_eval_loss,
            "test_ppl": avg_eval_ppl,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": (
                scaler.state_dict() if train_config.fp16_training else None
            ),
        }
        ckpt_handler.save(checkpoint)

    if train_config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyLLM Training help")
    parser.add_argument(
        "-m", "--model_type", type=str, help="Type of the model", required=True
    )
    args = parser.parse_args()
    run(model_type=args.model_type)
