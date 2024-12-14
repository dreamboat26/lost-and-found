import random
from typing import Generator, List, Tuple

import datasets
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Sampler
from transformers import PreTrainedTokenizer


class BatchSamplerSimilarLength(Sampler):
    def __init__(
        self,
        dataset_iterator: datasets.Dataset,
        batch_size: int,
        seq_len: int,
        shuffle: bool = True,
    ):
        """Initializer to load the dataset and sort as per the source sequences
        (german sequence) and prepare the indices in a bucketed manner where the
        sequences with similar lengths are grouped together.

        Args:
            dataset_iterator (datasets.Dataset): Dataset iterator.
            batch_size (int): Batch size to be used to compute upper limit of
                tokens.
            seq_len (int): Sequence length to be used to compute upper limit of
                tokens in a given batch. This will be directly correlated to
                available GPU memory.
            shuffle (bool, optional): Shuffle the dataset before sorting and
                after getting the buckets. Defaults to True.
        """
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.total_tokens_in_batch = self.batch_size * self.seq_len
        self.shuffle = shuffle

        self.indices = [
            (i, len(data["text"].split(" ")))
            for i, data in enumerate(dataset_iterator)
        ]

        if self.shuffle:
            random.shuffle(self.indices)

        sorted_indices = sorted(self.indices, key=lambda x: x[1])

        self.all_batch_idx = []
        single_batch_idx = []
        cummulative_token_len = 0

        for idx, token_len in sorted_indices:
            cummulative_token_len += token_len

            single_batch_idx.append(idx)

            if cummulative_token_len > self.total_tokens_in_batch:
                self.all_batch_idx.append(single_batch_idx.copy())
                single_batch_idx.clear()
                cummulative_token_len = 0

        if self.shuffle:
            random.shuffle(self.all_batch_idx)

    def __iter__(self) -> Generator[int, int, int]:
        """
        Function will fetch list of indices to be used to generate a batch.

        Yields:
            List[int]: Yields list of indices for batch generation.
        """
        for batch_idx in self.all_batch_idx:
            random.shuffle(batch_idx)
            yield batch_idx

    def __len__(self) -> int:
        """Function to get the total number of batches which can be generated.

        Returns:
            int: Number of batches from the given dataset.
        """
        return len(self.all_batch_idx)


class DatasetHelper:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        seq_len: int,
        max_seq_len: int,
        num_workers: int,
        persistent_workers: bool,
        split: str = "train",
    ):
        """
        Construct a dataset loader within this class.

        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer instance.
            batch_size  (int): Batch size to be used to compute upper limit of
                tokens.
            seq_len (int): Sequence length to be used to compute upper limit of
                tokens in a given batch. This will be directly correlated to
                available GPU memory.
            max_seq_len (int): Maximum sequence length to preserve. If the
                sequence is larger than max_seq_len then it will be clipped to
                max_seq_len. If it is shorter than this than it will be padded
                to maximum sequence length of the batch.
            num_workers (int): Number of workers for dataset loader.
            persistent_workers (bool): Use persistent_worker in torch dataloader.
            split (str, optional): Dataset split. Either "train" or "validation".
                Defaults to "train".

        Raises:
            RuntimeError: If unsupported split is provided.
        """
        if split not in ["train", "validation"]:
            raise RuntimeError(
                f"Split for dataloader shall be from 'train' or 'validation' only."
            )
        data_loader = load_dataset("roneneldan/TinyStories", split=split)

        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        batch_sampler = BatchSamplerSimilarLength(
            data_loader, batch_size, seq_len, shuffle=True
        )
        self.dataloader = DataLoader(
            data_loader,
            batch_sampler=batch_sampler,
            collate_fn=self.collate_batch,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

    def collate_batch(self, batch_data: List) -> Tuple[torch.Tensor]:
        """Function to tokenize sequences and prepare the mask for training.

        Args:
            batch_data (List): List of Tuple of source and target sequences.

        Returns:
            Tuple[torch.Tensor]: Tuple of source token, target token, source
                mask, target mask and target labels.
        """
        input_ids, attention_mask, labels = [], [], []

        for data in batch_data:
            text = data["text"]
            tokenized_data = self.tokenizer(
                text, max_length=self.max_seq_len, truncation=True
            )
            tokenized_input_ids = tokenized_data["input_ids"]
            tokenized_attn_mask = tokenized_data["attention_mask"]

            input_ids.append(
                torch.tensor(tokenized_input_ids, dtype=torch.int32)
            )
            labels.append(
                torch.tensor(
                    tokenized_input_ids[1:] + [self.pad_token_id],
                    dtype=torch.int32,
                )
            )
            attention_mask.append(
                torch.tensor(tokenized_attn_mask, dtype=torch.int32)
            )
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        attention_mask = pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels = pad_sequence(
            labels, batch_first=True, padding_value=self.pad_token_id
        )
        labels = labels.to(torch.long)
        return input_ids, attention_mask, labels

    def get_loader(self) -> DataLoader:
        """
        Get instance of dataloader.

        Returns:
            DataLoader: Dataloader based on split.
        """
        return self.dataloader


if __name__ == "__main__":
    from models.helper import train_config_factory
    from utils.misc import get_tokenizer

    model_type = "gpt"
    train_config = train_config_factory(model_type)
    tokenizer = get_tokenizer(model_type)

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

    for batch_idx, (input_ids, attn_mask, labels) in enumerate(valid_loader):
        print(f"Input ids: {input_ids}")
        print(f"Attention mask: {attn_mask}")
        print(f"Labels: {labels}")
        print(f"Input ids shape: {input_ids.shape}")
        print(f"Attention mask shape: {attn_mask.shape}")
        print(f"Labels shape: {labels.shape}")
        break
