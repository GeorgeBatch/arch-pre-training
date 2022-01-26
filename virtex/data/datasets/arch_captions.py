from collections import defaultdict
import json
import os
from typing import Dict, List

import cv2
from torch.utils.data import Dataset


class ArchCaptionsDatasetRaw(Dataset):
    r"""
    A PyTorch dataset to read ARCH dataset and provide it completely
    unprocessed. This dataset is used by various task-specific datasets
    in :mod:`~virtex.data.datasets` module.

    Args:
        data_root: Path to the ARCH dataset root directory.
        source: Name of ARCH source to read. One of ``{"pubmed", "books", "both"}``. Default value: "both".
        split:  Name of ARCH split to read. One of ``{"train", "val", "all"}``.
    """

    def __init__(self, data_root: str, source: str='both', split: str=''):
        allowed_source_values = ['pubmed', 'books', 'both']
        assert source in allowed_source_values, f"source should be one of {allowed_source_values}"
        allowed_split_values = ['train', 'val', 'all']
        assert split in allowed_split_values, f"split should be one of {allowed_split_values}"

        # Get path to the annotation file
        captions = json.load(
            open(os.path.join(data_root, "annotations", f"captions_{split}.json"))
        )

        # Collect list of uuids and file paths for each caption
        captions_to_uuids: Dict[str, List[str]] = defaultdict(list)
        captions_to_image_filepaths: Dict[str, List[str]] = defaultdict(list)
        captions_to_intids: Dict[str, List[int]] = defaultdict(list)
        for idx, ann in captions.items():
            if (source == "both") or (source == ann['source']):
                # if source="both", then no filtering needed
                # if source is one of the ["books", "pubmed"], LHS=False, RHS will filter the needed captions

                # annotation file contains a path relative to the `data_root`
                absolut_path = f"{data_root}/{ann['path']}"

                # make a check that the image exist before adding its `uuid` or `path`
                assert os.path.exists(absolut_path), f"{absolut_path} does not exist!"

                captions_to_image_filepaths[ann['caption']].append(absolut_path)
                # uuid (string); intid (int)
                captions_to_uuids[ann['caption']].append(ann['uuid'])
                captions_to_intids[ann['caption']].append(int(idx))
        #print(captions_per_image)

        # Keep all annotations in memory. Make a list of tuples, each tuple
        # is ``(list[image_id], list[file_path], captions)``.
        self.instances = [
            (captions_to_intids[caption], captions_to_image_filepaths[caption], caption)
            for caption in captions_to_image_filepaths.keys()
        ]

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx: int):
        image_ids, image_paths, caption = self.instances[idx]

        # shape: (height, width, channels), dtype: uint8
        images = [cv2.imread(image_path) for image_path in image_paths]
        # cv2.imread loads images in BGR (blue, green, red) order
        images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]

        return {"image_ids": image_ids, "images": images, "caption": caption}
