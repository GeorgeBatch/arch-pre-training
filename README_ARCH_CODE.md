# Understanding all code involved in VirTex pre-training

This document is my attempt to connect all the files involved in VirTex
pre-training in some systematic fashion and understand which of them need to be modified in order to pre-train models on the ARCH dataset.

## Building Vocabulary

### COCO

`scripts/build_vocabulary.py` is the main script for building COCO vocabulary.
It uses the original file with the train set of COCO captions:  `datasets/coco/annotations/captions_train2017.json`

### ARCH

1. Create one file with all the captions I want to use in the training set during pre-training on ARCH: `datasets/arch/annotations/captions_train.json`
2. Make `scripts/build_vocabulary_arch.py` for building ARCH vocabulary: remove 
duplicate captions since they occur when there is more than one image in the bag (figure) with the same caption. All images in a bag will be presented together with the corresponding caption so there is no reason to put more emphases on figures with multiple images than on figures with one image.

**TODO:** understand the sentence length - should I use the default 30 words?

## Pre-training

`scripts/pretrain_virtex.py` is the main script for running the pre-training 
process. It is very generic and does not need to be changed at all, instead 
the configuration `.yml` file needs to be changed and other code files need 
to be extended.

It starts with importing standard Python libraries. But also 
utilises custom classes and functions from this repository.

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

----
### 1. Config

`Config` class from `virtex/config.py` takes a `config_file` configuration 
file path as its main __init__ argument 
and an optional argument of the `override_list`. The default values set in 
`virtex/config.py` can be overridden with the `.yml` file so `virtex/config.py` does not need to be changed. See http://kdexd.xyz/virtex/virtex/config.html for full description.

**TODO:**
1. make a configuration `.yml` file to pass as an argument to the 
`Config` class through the parser.

#### Config Changes

**DATA**
* **ROOT**: "datasets/coco" -> "datasets/ARCH"
* **TOKENIZER_MODEL**: "datasets/vocab/coco_10k.model" -> 
  "datasets/vocab/**arch_10k**.model"
* **MAX_CAPTION_LENGTH**: TODO - check if 30 is enough
* **IMAGE_TRANSFORM_TRAIN**: "horizontal_flip" -> "tensor_horizontal_flip"


**MODEL**
* **NAME**: "virtex" -> "arch". Reason: Names in 
  :class:`PretrainingDatasetFactory` factory match with names in 
  :class:`PretrainingModelFactory` because both use same config parameter 
  `MODEL.NAME` to create objects.

**TODO:** check all the model changes that are needed.

* **NAME.VISUAL**: "torchvision::resnet50" -> "torchvision::resnet18"
* **NAME.TEXTUAL**: H=1024 (VirTex) -> H=512 (ARCH); A:=H/64=8, and F:=4H=2048




----
### 2. PretrainingDatasetFactory

`PretrainingDatasetFactory` class from `virtex/factories.py` creates a 
PyTorch Dataset(s) for pretraining VirTex models. It uses `ImageTransformsFactory` class to create image transforms, put them in a list, and use `albumentations.Compose()` method to compose them into a single transform. For captioning task it uses `CaptioningDataset` (gets image transform passed down), which in turn uses `CocoCaptionsDataset` (raw extraction).

**TODO:**

1. Make a version of CocoCaptionsDataset for ARCH: `ArchCaptionsDatasetRaw` ✅ 
2. Make a version of CaptioningDataset for ARCH: `ArchCaptioningDatasetExtended` ✅ 
3. Change all the relevant fields in the `.yml` file.

For the second task, need to figure out a way to 
transform all the images in the batch with the same flip so that they can 
be given together with the caption. This is achieved by performing all other 
transforms beforehand, placing all images from the bag into a tensor and 
then applying the flip on the tensor. Since the `albumentations`-based 
augmentations require individual images as inputs, a new class 
`TensorHorizontalFlip` is defined in `virtex/data/transforms.py`. It 
overrides the image-flipping method of the `HorizontalFlip` class from 
VirTex but keeps the method for changing the caption accordingly.

This also required to
* add both classes (`ArchCaptionsDatasetRaw` and 
`ArchCaptioningDatasetExtended`) into `virtex/data/__init__.py`; and
* change the `PretrainingDatasetFactory` and 
`ImageTransformsFactory` classes in `virtex/factories.py`.
  * `ImageTransformsFactory` got an extra line defining 
    special `tensor_horizontal_flip` operation:
```Python
"tensor_horizontal_flip": partial(T.TensorHorizontalFlip, p=0.5)
```
  * `PretrainingDatasetFactory` gets
    * PRODUCTS dictionary extended with an option `"arch": vdata. ArchCaptioningDatasetExtended`
    * 

**Problem** `albumentations` transforms want to be given an image, not a batch 
in a tensor.

#### From ARCH paper

For all of our MIC, MTL and MTL+MIC models, we employ standard data augmentations
using [imaug](https://imgaug.readthedocs.io/en/latest/index.html):
1. custom random crop function (such that letters are not cropped out);
2. [resize](https://imgaug.readthedocs.io/en/latest/source/overview/size.html?highlight=resize#resize) and [re-scale](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_size.html?highlight=re-scale#imgaug.augmenters.size.Scale);
3. color jitter ([brightness](https://imgaug.readthedocs.io/en/latest/source/overview/imgcorruptlike.html?highlight=brightness#brightness), [contrast saturation](https://imgaug.readthedocs.io/en/latest/source/overview/imgcorruptlike.html?highlight=contrast%20saturation#contrast) and [hue](https://imgaug.readthedocs.io/en/latest/source/overview/color.html?highlight=hue));
4. [Gaussian](https://imgaug.readthedocs.io/en/latest/source/overview/arithmetic.html?highlight=gaussian%20noise#additivegaussiannoise) and/or [salt and pepper](https://imgaug.readthedocs.io/en/latest/source/overview/arithmetic.html?highlight=salt%20and%20pepper#saltandpepper) noise; and
5. [JPEG compression](https://imgaug.readthedocs.io/en/latest/source/overview/arithmetic.html?highlight=JPEG%20compression%20artifacts#jpegcompression) artifacts.

Everything except the custom cropping has already been implemented by VirTex authors
using [albumentations](https://albumentations.ai/) library. I also added custom
horizontal flip to be able to flip all images in the bag together with their caption
at the same time.

**TODO:** Understand **custom cropping** with imgaug that will preserve letters on the images. 

----
### 3. PretrainingModelFactory
`PretrainingModelFactory` class from `virtex/factories.py` creates a 
model directly from config file. Uses **MODEL.NAME**, **MODEL.NAME.VISUAL** ,
and **MODEL.NAME.TEXTUAL** as the main arguments to choose which of the models
to create. 

**TODO:**
* Understand which parameters to put into config to have the same visual backbone and textual head as in the paper.
* Make any changes to the class if needed.

The model for ARCH dataset is the same as the main `VirTexModel` so we just 
set it in `PretrainingModelFactory` class.

Changes:
* Add to "arch" to PRODUCTS since **MODEL.NAME** we set is "arch".
    ```python
    "arch": vmodels.VirTexModel
    ```
* Add "arch" to the list of model names in:
```python
        if _C.MODEL.NAME in {"arch", "virtex", "captioning", "bicaptioning"}:
            kwargs = {
                "sos_index": _C.DATA.SOS_INDEX,
                "eos_index": _C.DATA.EOS_INDEX,
                "decoder": CaptionDecoderFactory.from_config(_C),
            }
```

`PretrainingModelFactory` uses 2 factories from the same file:
* `VisualBackboneFactory` class; and
* `TextualHeadFactory` class.

They are both fully specified by their respective names in the config file:
**MODEL.NAME.VISUAL** and **MODEL.NAME.TEXTUAL**.

**TODO:**
1. check if ResNet-18 works 
2. add batch-normalization layer to it at the front

----
### 4. OptimizerFactory
`OptimizerFactory` class from `virtex/factories.py` creates an optimizer directly from config

**TODO:**
1. Understand which parameters to put into config to have the same optimization as in the paper.
2. Add Adam optimizer as an option

----
### 5. LRSchedulerFactory
`LRSchedulerFactory` class from `virtex/factories.py` creates a learning-rate
scheduler directly from config. All schedulers have a built-in LR warmup schedule before actual LR scheduling (decay) starts.

**TODO:** understand which parameters to put into config to have the same lr schedule as in the paper.

----
### 6. The rest
The rest of the custom imports seem to be fine and do not need any input.
