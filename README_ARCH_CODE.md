# Understanding all of the code involved in VirTex pre-training

This document is my attempt to connect all the files involved in VirTex
pre-training in some systematic fashion and understand which of them need to be modified in order to pre-train models on the ARCH dataset.

## Building Vocabulary

**COCO:**

`scripts/build_vocabulary.py` is the main script for building COCO vocabulary.
It uses the original file with the train set of COCO captions:  `datasets/coco/annotations/captions_train2017.json`

**ARCH:**

1. Create one file with all the captions I want to use in the training set during pre-training on ARCH: `datasets/arch/annotations/captions_train.json`
2. Make `scripts/build_vocabulary_arch.py` for building ARCH vocabulary: remove dulplicate captions since they occur when there is more than one image in the bag (figure) with the same caption. All images in a bag will be presented together with the corresponding caption so there is no reason to put more emphases on figures with multiple images than on figures with one image.

TODO: understand the senetence length - should I use the default 30 words?

## Pre-training

`scripts/pretrain_virtex.py` is the main script for running the pre-training process. It starts with importing standard Python libraries. But also utilises custom classes and functions from this repository.

```Python
from virtex.config import Config  # 1
from virtex.factories import (   
    PretrainingDatasetFactory,    # 2
    PretrainingModelFactory,      # 3
    OptimizerFactory,             # 4
    LRSchedulerFactory,           # 5
)
from virtex.utils.checkpointing import CheckpointManager
from virtex.utils.common import common_parser, common_setup, cycle
import virtex.utils.distributed as dist
from virtex.utils.timer import Timer
```

1. `Config` class from `virtex/config.py` takes a `config_file` configuration file path as its main __init__ argument and an optional argument of the `override_list`. The default values set in `virtex/config.py` can be overridden with the `.yml` file so `virtex/config.py` does not need to be changed. See http://kdexd.xyz/virtex/virtex/config.html for full description.

**TODO:** make a configuration `.yml` file to pass as an argument to the `Config` class through the parser.

2. `PretrainingDatasetFactory` class from `virtex/factories.py` creates a PyTorch Dataset(s) for pretraining VirTex models. It uses `ImageTransformsFactory` class to create image transforms, put them in a list, and use `albumentations.Compose()` method to compose them into a single transform. For captioning task it uses `CaptioningDataset` (gets image transform passed down), which in turn uses `CocoCaptionsDataset` (raw extraction).

**TODO:**
* Make a version of CocoCaptionsDataset for ARCH ✅
* Make a version of CaptioningDataset for ARCH (figure out a way to transform all the images in the batch with the same flip so that they can be given together with the caption) ⏳

**Problem** Albumantation transforms want to be given an image, not a batch in a tensor.

**Idea**: do not put HorizontalFlip transform into the config file - do it separately.

3. `PretrainingModelFactory` class from `virtex/factories.py` creates a model directly from config file.

**TODO:** understand which parameters to put into config to have the same visual backbone and textual head as in the paper.

4. `OptimizerFactory` class from `virtex/factories.py` creates an optimizer directly from config

**TODO:** understand which parameters to put into config to have the same optimization as in the paper.

5. `LRSchedulerFactory` class from `virtex/factories.py` creates an lr sheduler directly from config. All schedulers have a built-in LR warmup schedule before actual LR scheduling (decay) starts.

**TODO:** understand which parameters to put into config to have the same lr schedule as in the paper.

6. The rest of the custom iports seem to be fine and do not need any input.
