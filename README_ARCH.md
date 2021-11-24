# Training VirTex on ARCH dataset

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

## Data Exploration

From original paper:
"ARCH contains 11,816 bags and 15,164 images in total.
Figure 4c shows a more detailed breakdown by the number of bags according to the number of images within the bag, with the smallest bag size being 1 (9,772 samples) and the largest bag size being 9 for which we have only 7 samples."

TODO: These numbers need to be recomputed.


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

"For all of our MIC, MTL and MTL+MIC models, we employ standard data augmentations using imaug: custom random crop function such that letters are not cropped out); resize and re-scale; color jitter (brightness, contrast satura- tion and hue); Gaussian and/or salt and pepper noise, and JPEG compression artifacts."

* resnet50 -> resnet18
* Fast Food Transform
* ? CLAM-slyle (not all of the layers)
* BatchNorm at the Input

###

2.
