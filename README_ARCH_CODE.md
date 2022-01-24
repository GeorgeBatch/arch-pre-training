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

1. `Config` class from `virtex/config.py` takes a `config_file` configuration file path as its main __init__ argument and an optional argument of the `override_list`. The default values set in `virtex/config.py` can be overridden with the `.yml` file so `virtex/config.py` does not need to be changed.

**TODO:** make a configuration `.yml` file to pass as an argument to the `Config` class through the parser.
