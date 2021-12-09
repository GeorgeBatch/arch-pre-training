import random
from typing import Callable, Dict, List

import albumentations as alb
import numpy as np
import torch
from torch.utils.data import Dataset

from virtex.data.tokenizers import SentencePieceBPETokenizer
from virtex.data import transforms as T
from .coco_captions import ArchCaptionsDatasetRaw


class ArchCaptioningDatasetExtended(Dataset):
    r"""
    A dataset which provides image-caption (forward and backward) pairs from
    a ARCH Captions annotation file. This is used for pretraining tasks which
    use captions - bicaptioning, forward captioning and token classification.

    Args:
        data_root: Path to dataset directory containing images and annotations.
        source: Name of ARCH source to read. One of ``{"pubmed", "books", "both"}``.
            "both" option results in a concatenation of the datasets from "pubmed" and "books"
        split: Name of ARCH split to read. One of ``{"train", "val", "all"}``.
        tokenizer: Tokenizer which maps word tokens to their integer IDs.
        image_transform: List of image transformations, from either
            `albumentations <https://albumentations.readthedocs.io/en/latest/>`_
            or :mod:`virtex.data.transforms`.
        max_caption_length: Maximum number of tokens to keep in caption tokens.
            Extra tokens will be trimmed from the right end of the token list.
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        tokenizer: SentencePieceBPETokenizer,
        source: str="both",
        image_transform: Callable = T.DEFAULT_IMAGE_TRANSFORM,
        # TODO: add a default random transform (to be applied to a batch=bag of images)
        # random_image_transform: Callable = ,
        max_caption_length: int = 30,
    ):
        self._dset = ArchCaptionsDatasetRaw(data_root=data_root, source=source, split=split)
        self.image_transform = image_transform
        self.caption_transform = alb.Compose(
            [
                T.NormalizeCaption(),
                T.TokenizeCaption(tokenizer),
                T.TruncateCaptionTokens(max_caption_length),
            ]
        )
        self.padding_idx = tokenizer.token_to_id("<unk>")

    def __len__(self):
        return len(self._dset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        # keys: {"image_id", "image", "captions"}
        instance = self._dset[idx]
        image_ids, images, caption = (
            instance["image_ids"],
            instance["images"],
            instance["caption"],
        )

        # List[int] -> np.array of shape (len(image_ids), )
        image_ids = np.array(image_ids)
        # (len(image_ids), ) -> (len(image_ids), 1)
        image_ids = image_ids.reshape((image_ids.shape[0], 1))

        # Transform image-caption pairs.
        #    All images need to be resized to the same size to put them into a tensor.
        # Pass in caption to image_transform due to paired horizontal flip.
        # Caption won't be tokenized/processed here.
        image_caption_pairs = [self.image_transform(image=image, caption=caption) for image in images]

        # split images and captions
        images = [image_caption["image"] for image_caption in image_caption_pairs]
        captions = [image_caption["caption"] for image_caption in image_caption_pairs]
        assert all([captions[0] == captions[i] for i in range(len(captions))]), \
            "The same transformation should be performed on all images in a bag => single version of the caption should appear"
        caption_tokens = self.caption_transform(caption=captions[0])["caption"]

        # Convert each image from HWC to CHW format and convert to tensors:
        #   Transforms expect to receive tensors in (B, C, H, W) shape
        images = [np.transpose(image, (2, 0, 1)) for image in images] # [(Channel, Height, Width), ..., ] Bag Size times
        images = [torch.tensor(image, dtype=torch.float) for image in images]

        return {
            "image_ids": torch.tensor(image_ids, dtype=torch.long), # ((Bag size, 1)
            "images": torch.stack(images, dim=0), # (Bag size, Channel, Height, Width)
            "caption_tokens": torch.tensor(caption_tokens, dtype=torch.long),
            "noitpac_tokens": torch.tensor(caption_tokens, dtype=torch.long).flip(0),
            "caption_lengths": torch.tensor(len(caption_tokens), dtype=torch.long),
        }

    def collate_fn(
        self, data: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:

        # Pad `caption_tokens` and `masked_labels` up to this length.
        caption_tokens = torch.nn.utils.rnn.pad_sequence(
            [d["caption_tokens"] for d in data],
            batch_first=True,
            padding_value=self.padding_idx,
        )
        noitpac_tokens = torch.nn.utils.rnn.pad_sequence(
            [d["noitpac_tokens"] for d in data],
            batch_first=True,
            padding_value=self.padding_idx,
        )
        return {
            "image_id": torch.stack([d["image_ids"] for d in data], dim=0),
            "image": torch.stack([d["images"] for d in data], dim=0),
            "caption_tokens": caption_tokens,
            "noitpac_tokens": noitpac_tokens,
            "caption_lengths": torch.stack([d["caption_lengths"] for d in data]),
        }
