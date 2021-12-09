# Training VirTex models on ARCH dataset

## Download the Dataset

The dataset should be placed into `./datasets/ARCH/`

You can download the 2 parts of the ARCH dataset (`books_set` and `pubmed_set`) from from https://warwick.ac.uk/fac/cross_fac/tia/data/arch.


Alternatively, you can do it from the command line. The script uses `wget` to download the dataset, `unzip` to inflate the `.zip` files, deletes the `__MACOSX` folders and `.zip` archives.

```{shell}
cd ./arch/
bash 0-download-arch.sh
```

## Data Exploration (Dataset Changed!)

**Annotation files:**
* Books Set: ([datasets/ARCH/books_set/captions.json](datasets/ARCH/books_set/captions.json))
* PubMed Set: ([datasets/ARCH/books_set/captions.json](datasets/ARCH/books_set/captions.json))

**Code:**
* [arch/1-ARCH-Data-Exploration.ipynb](arch/1-ARCH-Data-Exploration.ipynb) contains my code for data exploration

**From the original paper:**
"ARCH contains 11,816 bags and 15,164 images in total.
Figure 4c shows a more detailed breakdown by the number of bags according to the number of images within the bag, with the smallest bag size being 1 (9,772 samples) and the largest bag size being 9 for which we have only 7 samples."

**Changes:**
The dataset available on the web is smaller than the one described in the paper.
I decided to recompute the statistics of the dataset.
See [arch/1-ARCH-Data-Exploration.ipnb](arch/1-ARCH-Data-Exploration.ipynb) for .

**Summary:**
* Total Images. Books: 4270 `.png` images; PubMed: 3309 (3272 `.jpg`, 37 `.png`) images. Total: 7579 images.
* Books set annotation file contains 35 captions w/o images; Pubmed captions ([datasets/ARCH/pubmed_set/captions.json](datasets/ARCH/pubmed_set/captions.json)) map one-to-one to the downloaded images.
* Books set has 2 ways of groupping images into bags. By `figure_id` and by `caption` fields of its annotation file. Pubmed set does not have the `figure_id` field so images are groupped into bags using capitons.
* Books set annotation file contains 76 captions that correspond to 2 figure ids and 1 caption corresponds to 3 figure ids. This means that there are 78 more unique figure ids than captions. **This is suspected to be a mistake and should be addressed.**

**Decisions:**
* Ignored the entries of the annotation files with missing images.
* Parsed the file extensions on the fly in the dataset classes.

**TODO:**
* Decide what to do with captions corresponding to multiple figures in the Books Set.

**Full Version:**
[README_ARCH_DATA.md](README_ARCH_DATA.md) contains everything found during the data exploration

## Creating annotation files

**Code:**
* [arch/2-ARCH-All-Annotations-In-One-File.ipynb](arch/2-ARCH-All-Annotations-In-One-File.ipynb)
* [arch/3-ARCH-Train-Val-Split.ipynb](arch/3-ARCH-Train-Val-Split.ipynb)


Run all cells in both notebooks to create the annotation files:
* [datasets/ARCH/annotations/captions_all.json](datasets/ARCH/annotations/captions_all.json)
* [datasets/ARCH/annotations/captions_train.json](datasets/ARCH/annotations/captions_train.json)
* [datasets/ARCH/annotations/captions_val.json](datasets/ARCH/annotations/captions_val.json)

The original annotation files provided with the dataset included references to some 35 non-existent images. They were also not suitable for the dataset classes from VirTex. The original annotation files can be found here:
* [datasets/ARCH/books_set/captions.json](datasets/ARCH/books_set/captions.json)
* [datasets/ARCH/pubmed_set/captions.json](datasets/ARCH/pubmed_set/captions.json)


The annotation files created preserve the `figure_id` and `letter` columns from the `books_set`. For the instances from the `pubmed_set`, these columns are set to `None`.


## Raw Dataset Class

**VirTex Code (original)**
* [virtex/data/datasets/coco_captions.py](virtex/data/datasets/coco_captions.py) provides a "PyTorch dataset to read COCO Captions dataset and provide it completely unprocessed" (`CocoCaptionsDataset`). It needs to be changed to account for the differences between the COCO and the ARCH datasets. COCO has one of more captions per image, while ARCH has a single caption per one or more images. The directory structure is also different.

**ARCH Code (extended by me)**
* [virtex/data/datasets/arch_captions.py](virtex/data/datasets/arch_captions.py) contains my class analogous to `CocoCaptionsDataset` called `ArchCaptionsDatasetRaw`. [arch/4-ARCH-Dataset-Class-Raw.ipynb](arch/4-ARCH-Dataset-Class-Raw.ipynb) shows examples of its basic usage.

## Building Vocabulary

**Code:**
* [scripts/build_vocabulary_arch.py](scripts/build_vocabulary_arch.py) is an adapted version of the original [scripts/build_vocabulary.py](scripts/build_vocabulary.py) file from VirTex. The only differences there are to accommodate the differences between the structures of the annotation files and multiple occurrences of captions.

**Duplicate captions are removed before passing the list of captions to the tokenizer since for each bag of images its caption is presented only once.**

Following J. Gamper *et al.*, all other parameters are kept as defaults from VirTex.

**Run from the root directory:**
```
mkdir datasets/vocab/
python scripts/build_vocabulary_arch.py
```

It results in the creation of:
* `datasets/vocab/arch_10k.model`
* `datasets/vocab/arch_10k.vocab`

## Extended Dataset Class

**VirTex Code (original)**
* [virtex/data/datasets/captioning.py](virtex/data/datasets/captioning.py) provides an extended PyTorch Dataset class (`CaptioningDataset`) which specifies
  1. The caption selected at random (COCO has multiple captions per image)
  2. Paired Image-Caption Augmentations to be used (horizontal flip combined with "left" <-> "right" switch)
  3. Text tokenisation (after augmentations)
  4. Collate function to put the items into batches

**ARCH Code (extended by me)**
* [virtex/data/datasets/captioning.py](virtex/data/datasets/captioning.py) provides the extended version of the Dataset class analogous to `CaptioningDataset` called `ArchCaptioningDatasetExtended`. [arch/5-ARCH-Dataset-Class-Extended.ipynb](arch/5-ARCH-Dataset-Class-Extended.ipynb) shows examples of its basic usage. It requires to specify a tokeniser model, e.g. "`datasets/vocab/arch_10k.model"`.

**Differences to the VirTex Code:**
ARCH has one or more images per caption. **I assume that only all images together will contain enough information to correspond to a caption.** This means that:
1. There is no need to select a caption at random (only one present anyway)
2. Images from one bag should be given together as one batch with a single caption as its label.
3. All transforms (both fixed and random) except for the "flips" can be different for different images in the bag since only the "flips" are connected to the changes in captions. After all transforms are performed, the images should be put in a single tensor of shape (Batch Size = Bag Size, C, Height, Width).
4. If the flip is performed, it should be the same flip so that the caption is adjusted in the same way.


## Libraries used in the ARCH paper

* Extracting images and captions from pubmed: https://github.com/titipata/pubmed_parser
* PDF-Figures 2.0 for extracting figures from 10 textbooks https://github.com/allenai/pdffigures2
* Intrinsic dimensionality estimation https://github.com/jgamper/intrinsic-dimensionality
* Desai et al. [12] code available at https://github.com/kdexd/virtex
* Mormont et al. [46] code used for multi-task learning available at https://github.com/waliens/multitask-dipath.

## Changes needed to train on ARCH dataset

TODO: understand which changes need to be made to the config file.

### Model

"For each of the tasks, as well as for all of our experiments in the paper, we used ResNet-18 [22] as an encoder with its parameters wrapped in Fast-food transform [38] for obtaining random sub-spaces required for intrinsic dimensionality estimation. All encoders were randomly initialised."

"For both models, we modify ResNet-18 to have the Batch-norm layer as an input layer."

"the only difference with that of Desai et al. [12] is that of working with bags of image instances instead of a single instance"

"For all MIC model training experiments, we set the hyper-paremeters, tokenization and training details according to Desai et al. [12] and their publicly available code, with a few exceptions as follows: H is set to 512, which also determines the width of each of the transformer layers and the number of attention heads; the batch size is set to 32 images or less irrespective of the bag sizes due to computational restraints, which are pre-computed before every epoch after re-shuffling the dataset indices. Finally, for the ease of training, we switched to a ADAM optimizer with a default learning rate of 1e-3 and an early stopping set to a patience of a held-out set validation loss of 10."

"For all of our MIC, MTL and MTL+MIC models, we employ standard data augmentations using [imaug](https://imgaug.readthedocs.io/en/latest/): custom random crop function such that letters are not cropped out); resize and re-scale; color jitter (brightness, contrast saturation and hue); Gaussian and/or salt and pepper noise, and JPEG compression artifacts."

* resnet50 -> resnet18
* Fast Food Transform
* ? CLAM-slyle (not all of the layers)
* BatchNorm at the Input



## Instructions from VirTex webpage

Source: http://kdexd.xyz/virtex/virtex/usage/pretrain.html

### How to train your VirTex model?

We provide training scripts for all type of VirTex models from the paper; including our best-performing model and other ablations. Our training jobs are specified by config files (YAML). Execute all commands from project root to use the provided config files.

#### Training the base VirTex model

Train the base VirTex model with ResNet-50 visual backbone; and a textual head with `L = 1, H = 1024` using all default optimization hyperparameters.

```{shell}
python scripts/pretrain_virtex.py \
    --config configs/_base_bicaptioning_R_50_L1_H1024.yaml \
    --num-gpus-per-machine 8 \
    --cpu-workers 4 \
    --serialization-dir /tmp/VIRTEX_R_50_L1_H1024
    # Default: --checkpoint-every 2000 --log-every 20
```

Training job will save checkpoints, tensorboard logs (loss curves and metrics), and back up the config in `--serialization-dir`. Use `tensorboard --logdir <serialization_dir>` to view training curves, validation metrics etc. directly on tensorboard.

We recommend training with 8 GPUs on the same machine, although training with multiple GPUs across machines (see: `--num-machines` and `--machine-rank`), single GPU (`--num-gpus-per-machine 1`) as well as CPU (`--num-gpus-per-machine 0`) is also supported. Using multiple GPUs for interactive debugging with PDB is not supported, as PDB and multiprocessing module do not play nice.

### Reproducing all VirTex ablations

To reproduce all ablations from the paper, replace the `--config` argument in above command with the following (all assumed to be relative to project root):

#### Pretraining Task Ablations

1. Bicaptioning: configs/task_ablations/bicaptioning_R_50_L1_H2048.yaml
2. Forward Captioning: configs/task_ablations/captioning_R_50_L1_H2048.yaml
3. Token Classification: configs/task_ablations/token_classification_R_50.yaml
4. Multilabel Classification: configs/task_ablations/multilabel_classification_R_50.yaml
5. Masked Language Modeling: configs/task_ablations/masked_lm_R_50_L1_H2048.yaml

#### Transformer Size Ablations

1. Width (H = 512): configs/width_ablations/bicaptioning_R_50_L1_H512.yaml
2. Width (H = 768): configs/width_ablations/bicaptioning_R_50_L1_H768.yaml
3. Width (H = 1024): configs/width_ablations/bicaptioning_R_50_L1_H1024.yaml
4. Width (H = 2048): configs/width_ablations/bicaptioning_R_50_L1_H2048.yaml
5. Depth (L = 1): configs/depth_ablations/bicaptioning_R_50_L1_H1024.yaml
6. Depth (L = 2): configs/depth_ablations/bicaptioning_R_50_L2_H1024.yaml
7. Depth (L = 3): configs/depth_ablations/bicaptioning_R_50_L3_H1024.yaml
8. Depth (L = 4): configs/depth_ablations/bicaptioning_R_50_L4_H1024.yaml

#### Backbone Ablations

1. ResNet-50: configs/backbone_ablations/bicaptioning_R_50_L1_H1024.yaml
2. ResNet-50 w2x: configs/backbone_ablations/bicaptioning_R_50W2X_L1_H1024.yaml
3. ResNet-101: configs/backbone_ablations/bicaptioning_R_101_L1_H1024.yaml

```text
Note
Pretraining Task Ablations (1), Transformer Size Ablations (3 and 5) and Backbone Ablations (1) are all the same exact model.
```
