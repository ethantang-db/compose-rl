# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Dataloader builders."""

from functools import partial
from typing import Any, Callable, Iterable, Sequence, Union

from streaming import Stream, StreamingDataLoader, StreamingDataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from compose_rl.data.messages_data import (
    MessagesStreamingDataset,
    messages_dataset_collate_fn,
)
from compose_rl.data.offline_data import (
    OfflineStreamingDataset,
    offline_dataset_collate_fn,
    offline_dataset_collate_fn_test,
)
from compose_rl.data.preference_data import (
    FinegrainedPreferenceStreamingDataset,
    PairwisePreferenceStreamingDataset,
    finegrained_preference_dataset_collate_fn,
    pairwise_preference_dataset_collate_fn,
)
from compose_rl.data.prompt_data import (
    PromptStreamingDataset,
    prompt_dataset_collate_fn,
)

__all__ = [
    'build_finegrained_preference_dataloader',
    'build_pairwise_preference_dataloader',
    'build_prompt_dataloader',
    'build_messages_dataloader',
    'build_offline_dataloader',
]


import torch
from composer.core.data_spec import DataSpec, _split_mapping
from composer.core.types import Batch


def _qwen3_vl_get_num_samples_in_batch(batch: Batch) -> int:
    # This is the only source of truth for how big the batch is.
    return batch['input_ids'].shape[0]


def _qwen3_vl_split_batch(batch: Batch, microbatch_size: int | float) -> Sequence[Batch]:
    if isinstance(microbatch_size, float):
        raise NotImplementedError('Float microbatch size is not supported for Qwen3 VL models.')

    if microbatch_size >= batch['input_ids'].shape[0]:
        return [batch]

    # first clone the batch because somewhere they use view and we want it to be a copy
    cloned_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    pixel_values = cloned_batch['pixel_values']
    del cloned_batch['pixel_values']  # we will split this one separately

    # find the number of patches per image (pixel_value), this comes from the image_grid_thw tensor
    # which is a multiple of the temporal dimension, width, and height
    patches_per_image = cloned_batch['image_grid_thw'].prod(dim=1)

    # split other items by mapping
    new_batches = _split_mapping(cloned_batch, microbatch_size)

    # split pixel_value by the number of patches per image per micro batch
    microbatch_num_patches = 0
    batch_idx = 0
    microbatch_idx = 0
    cur_pixel_values_patch_idx = 0
    while batch_idx < len(batch['input_ids']):
        for _ in range(microbatch_size):
            microbatch_num_patches += patches_per_image[batch_idx]
            batch_idx += 1
            if batch_idx >= len(batch['input_ids']):
                break

        new_batches[microbatch_idx]['pixel_values'] = pixel_values[
            cur_pixel_values_patch_idx:cur_pixel_values_patch_idx + microbatch_num_patches]
        cur_pixel_values_patch_idx += microbatch_num_patches
        microbatch_idx += 1
        microbatch_num_patches = 0
    return new_batches


def _qwen3_vl_get_num_tokens_in_batch(batch: Batch) -> dict[str, int]:
    text_tokens = batch['input_ids'].numel()
    image_tokens = batch['pixel_values'].shape[0]
    return text_tokens + image_tokens


class Qwen3VLDataSpec(DataSpec):
    """Data specification for Qwen3 VL models.

    Args:
        dataloader (DataLoader): The dataloader.
    """

    def __init__(self, dataloader: Union[Iterable, DataLoader], **kwargs: Any):
        super().__init__(
            dataloader=dataloader,
            get_num_samples_in_batch=_qwen3_vl_get_num_samples_in_batch,
            split_batch=_qwen3_vl_split_batch,
            get_num_tokens_in_batch=_qwen3_vl_get_num_tokens_in_batch,
            **kwargs,
        )


def get_qwen3_vl_data_spec(
    dl: Union[Iterable, DataLoader],
    dataset_cfg: dict[str, Any],
) -> Qwen3VLDataSpec:
    return Qwen3VLDataSpec(dataloader=dl)


def generate_dataloader_builder(
    dataset_cls: type[StreamingDataset],
    collate_fn: Callable,
) -> Callable:
    """Generates dataloader builder for a given dataset_cls and collate_fn."""

    def build_preference_dataloader(
        tokenizer: PreTrainedTokenizer,
        device_batch_size: int,
        dataset: dict[str, Any],
        drop_last: bool,
        num_workers: int,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        timeout: int = 0,
    ) -> DataLoader:
        """Builds a dataloader for prompt data.

        Args:
            tokenizer: the model's tokenizer.
            device_batch_size: batch size per device.
            dataset: the dataset configuration.
            drop_last: indicating if we should drop the last batch.
            num_workers: number of workers to use.
            pin_memory: indicating if we should pin memory.
            prefetch_factor: the prefetch factor.
            persistent_workers: indicating if we should use persistent workers.
            timeout: the timeout value.
        """
        dataset_cfg = dataset

        streams_dict = dataset_cfg.pop('streams', None)
        max_seq_len = dataset_cfg.get('max_seq_len', None)
        if max_seq_len is None:
            raise ValueError(
                'max_seq_len must be provided in the dataset configuration',
            )

        # Build streams
        streams = None
        if streams_dict is not None:
            streams = [Stream(**stream) for stream in streams_dict.values()]
        if issubclass(
            dataset_cls,
            MessagesStreamingDataset,
        ) and 'tokenizer' not in dataset_cfg:
            dataset_cfg['tokenizer'] = tokenizer

        streaming_dataset = dataset_cls(
            streams=streams,  # type: ignore
            batch_size=device_batch_size,  # type: ignore
            **dataset_cfg,
        )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        dataloader = StreamingDataLoader(
            streaming_dataset,
            collate_fn=partial(collate_fn, tokenizer, max_seq_len),
            batch_size=device_batch_size,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            timeout=timeout,
        )
        return get_qwen3_vl_data_spec(dataloader, {})

    return build_preference_dataloader


build_pairwise_preference_dataloader = generate_dataloader_builder(
    PairwisePreferenceStreamingDataset,
    pairwise_preference_dataset_collate_fn,
)

build_finegrained_preference_dataloader = generate_dataloader_builder(
    FinegrainedPreferenceStreamingDataset,
    finegrained_preference_dataset_collate_fn,
)

build_prompt_dataloader = generate_dataloader_builder(
    PromptStreamingDataset,
    prompt_dataset_collate_fn,
)

build_messages_dataloader = generate_dataloader_builder(
    MessagesStreamingDataset,
    messages_dataset_collate_fn,
)

build_offline_dataloader = generate_dataloader_builder(
    OfflineStreamingDataset,
    offline_dataset_collate_fn_test,
)
