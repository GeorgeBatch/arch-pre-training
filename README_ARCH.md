# Training VirTex models on ARCH dataset

## Data Exploration (Dataset Changed!)

From the original paper:
"ARCH contains 11,816 bags and 15,164 images in total.
Figure 4c shows a more detailed breakdown by the number of bags according to the number of images within the bag, with the smallest bag size being 1 (9,772 samples) and the largest bag size being 9 for which we have only 7 samples."

The dataset available on the web is smaller than the one described in the paper.
I decided to recompute the statistics of the dataset.
See `./arch/1-ARCH-Data-Exploration.ipnb` for implementation.

### Downloading the Dataset

The dataset should be placed into `./datasets/ARCH/`

You can download the 2 parts of the ARCH dataset (`books_set` and `pubmed_set`) from from https://warwick.ac.uk/fac/cross_fac/tia/data/arch.


Alternatively, you can do it from the command line. The script uses `wget` to download the dataset, `unzip` to inflate the `.zip` files, deletes the `__MACOSX` folders and `.zip` archives.

```{shell}
cd ./arch/
bash 0-download-arch.sh
```


### Books Set

Some images from the `books_set` may contain letters on top. This letters are there to understand which part of the caption refers to which the image when multiple images correspond to one caption.

* Total Images (`./datasets/ARCH/books_set/images/`): 4270
* Total Caption Rows (`./datasets/ARCH/books_set/captions.json`): 4305
* **Captions with missing images**: 35

All of the images in the `./datasets/ARCH/books_set/images/` directory have a corresponding caption, but not all captions have a corresponding image.

**This table was computed from captions.json:**

Bag Size | # Bags
-------- | ------
1        | 2720   
2        | 378
3        | 133
4        | 56
5        | 12
6        | 14
7        | 4
8        | 2
9        | 2

`# Bags` is calculated using `figure_id` field, not `caption` field - not sure if it's the right way.

* Total unique captions: 3241
* Total unique figure_ids: 3321

**Due to the missing images, the values need to be recomputed.**

Bag Size | # Bags | Difference
-------- | ------ | ---------
1        | 2688   | 32
2        | 378    |
3        | 132    | 1
4        | 56     |
5        | 12     |
6        | 14     |
7        | 4      |
8        | 2      |
9        | 2      |

`# Bags` is calculated using `figure_id` field, not `caption` field - not sure if it's the right way.


Total difference is 35 = 32x1 + 1x3 images.

* Total unique captions: 3210
* Total unique figure ids: 3288

TODO: understand why there is no 1-to-1 correspondence between captions and figure ids.

### PubMed Set

* Total Images (`./datasets/ARCH/pubmed_set/images/`): 3309
* Total Caption Rows (`./datasets/ARCH/pubmed_set/captions.json`): 3309
* Captions with missing images: 0

Bag Size | # Bags
-------- | ------
1        | 3270
2        | 11
3        | 2
4        | 0
5        | 1
6        | 1

* Total unique captions: 3285 = 3270 + 11 + 2 + 0 + 1 + 1

**Captions are not split into different images. There are no "A", "B", "C" parts in a caption. There are also no "A", "B", "C" labels on images. This means that images with the same caption can be put in a bag with the caption, but also can probably be given to the model one by one.**

### Together (only counting when images are available)

Bag Size | # Bags
-------- | ------
1        | 5958   
2        | 389
3        | 134
4        | 56
5        | 13
6        | 15
7        | 4
8        | 2
9        | 2

* Total Images (`./datasets/ARCH/*/images/`): 7579 = 3309 + 4270
* Total Captions


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
