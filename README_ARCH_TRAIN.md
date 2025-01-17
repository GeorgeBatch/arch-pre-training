This is a modified and slightly extended version of the setup information
provided in:
http://kdexd.xyz/virtex/virtex/usage/pretrain.html

# How to train your VirTex model?


We provide training scripts for all type of VirTex models from the paper;
including our best-performing model and other ablations.
Our training jobs are specified by config files (YAML).
Execute all commands from project root to use the provided config files.


## Training the base VirTex model

Train the base VirTex model with ResNet-50 visual backbone; and a textual head
with ``L = 1, H = 1024`` using all default optimization hyperparameters.

```shell
    python scripts/pretrain_virtex.py \
        --config configs/_base_bicaptioning_R_50_L1_H1024.yaml \
        --num-gpus-per-machine 8 \
        --cpu-workers 4 \
        --serialization-dir /tmp/VIRTEX_R_50_L1_H1024
        # Default: --checkpoint-every 2000 --log-every 20
```


Training job will save checkpoints, tensorboard logs (loss curves and metrics),
and back up the config in ``--serialization-dir``. Use ``tensorboard --logdir
<serialization_dir>`` to view training curves, validation metrics etc. directly
on tensorboard.

We recommend training with 8 GPUs on the same machine, although training with
multiple GPUs across machines (see: ``--num-machines`` and ``--machine-rank``),
single GPU (``--num-gpus-per-machine 1``) as well as CPU
(``--num-gpus-per-machine 0``) is also supported. Using multiple GPUs for
interactive debugging with PDB is not supported, as PDB and ``multiprocessing``
module do not play nice.

-------------------------------------------------------------------------------

## Reproducing all VirTex ablations

To reproduce all ablations from the `paper <https://arxiv.org/abs/2006.06666>`_,
replace the ``--config`` argument in above command with the following (all
assumed to be relative to project root):

### Pretraining Task Ablations

1. **Bicaptioning:** configs/task_ablations/bicaptioning_R_50_L1_H2048.yaml
2. **Forward Captioning:** configs/task_ablations/captioning_R_50_L1_H2048.yaml
3. **Token Classification:** configs/task_ablations/token_classification_R_50.yaml
4. **Multilabel Classification:** configs/task_ablations/multilabel_classification_R_50.yaml
5. **Masked Language Modeling:** configs/task_ablations/masked_lm_R_50_L1_H2048.yaml

### Transformer Size Ablations

1. **Width (H = 512):** configs/width_ablations/bicaptioning_R_50_L1_H512.yaml
2. **Width (H = 768):** configs/width_ablations/bicaptioning_R_50_L1_H768.yaml
3. **Width (H = 1024):** configs/width_ablations/bicaptioning_R_50_L1_H1024.yaml
4. **Width (H = 2048):** configs/width_ablations/bicaptioning_R_50_L1_H2048.yaml
5. **Depth (L = 1):** configs/depth_ablations/bicaptioning_R_50_L1_H1024.yaml
6. **Depth (L = 2):** configs/depth_ablations/bicaptioning_R_50_L2_H1024.yaml
7. **Depth (L = 3):** configs/depth_ablations/bicaptioning_R_50_L3_H1024.yaml
8. **Depth (L = 4):** configs/depth_ablations/bicaptioning_R_50_L4_H1024.yaml

### Backbone Ablations

1. **ResNet-50:** configs/backbone_ablations/bicaptioning_R_50_L1_H1024.yaml
2. **ResNet-50 w2x:** configs/backbone_ablations/bicaptioning_R_50W2X_L1_H1024.yaml
3. **ResNet-101:** configs/backbone_ablations/bicaptioning_R_101_L1_H1024.yaml


**Pretraining Task Ablations** (1), **Transformer Size Ablations** (3 and 5) and
**Backbone Ablations** (1) are all the same exact model.


## Training the base ARCH model

Train the base VirTex model with ResNet-50 visual backbone; and a textual head
with ``L = 1, H = 1024`` using all default optimization hyperparameters.


**CPU**:
```shell
python scripts/pretrain_virtex.py \
    --config configs/_arch_bicaptioning_R_18_L1_H512.yaml \
    --num-gpus-per-machine 0 \
    --cpu-workers 2 \
    --serialization-dir /tmp/VIRTEX_R_18_L1_H512
    # Default: --checkpoint-every 2000 --log-every 20
```

**GPU - DEBUGGING**:
```shell
python scripts/pretrain_virtex.py \
    --config configs/_arch_bicaptioning_R_18_L1_H512.yaml \
    --num-gpus-per-machine 1 \
    --cpu-workers 4 \
    --serialization-dir /tmp/VIRTEX_R_18_L1_H512
    # Default: --checkpoint-every 2000 --log-every 20
```

**GPU - FULL ON**:
```shell
python scripts/pretrain_virtex.py \
    --config configs/_arch_bicaptioning_R_18_L1_H512.yaml \
    --num-gpus-per-machine 4 \
    --cpu-workers 32 \
    --serialization-dir /tmp/VIRTEX_R_18_L1_H512
    # Default: --checkpoint-every 2000 --log-every 20
```