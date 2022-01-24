This is a modified and slightly extended version of the setup information
provided in:
http://kdexd.xyz/virtex/virtex/usage/setup_dependencies.html

# How to setup this codebase?

This codebase requires Python 3.6+ or higher. We recommend using Anaconda or
Miniconda. We walk through installation and data preprocessing here.


# Install Dependencies

For these steps to install through Anaconda (or Miniconda).

1. Install Anaconda or Miniconda distribution based on Python 3+ from their
   [downloads site](https://conda.io/docs/user-guide/install/download.html).

2. Clone the repository first.

If working on a remote server, you might need to first load your git
module:

```shell
module load git/2.33.1-GCCcore-11.2.0-nodocs
```

```shell
git clone https://www.github.com/GeorgeBatch/arch-pre-training
```

3. Create a conda environment and install all the dependencies.

If working on a remote server, you might need to first load your Anaconda
module:

```shell
module load Anaconda3/2020.11
eval "$(conda shell.bash hook)"
```


```shell
cd arch-pre-training
conda create -n virtex python=3.8 # kept the environment name the same
conda activate virtex
pip install -r requirements.txt
```

4. Install additional packages from Github.

```shell
conda activate virtex # just in case (if already done the previous step)

pip install git+git://github.com/facebookresearch/fvcore.git#egg=fvcore
pip install git+git://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
```

Install missing packages

```shell
conda install albumentations # augmentations
conda install sentencepiece # tokenization
conda install jupyter # to run in notebooks
```


5. Install this codebase as a package in development version.

    .. code-block:: shell

        python setup.py develop

Now you can ``import virtex`` from anywhere as long as you have this conda
environment activated.

-------------------------------------------------------------------------------


Setup Datasets
--------------

Datasets are assumed to exist in ``./datasets`` directory (relative to the
project root) following the structure specified below. COCO is used for
pretraining, and rest of the datasets (including COCO) are used for downstream
tasks. This structure is compatible when using
`Detectron2 <https://github.com/facebookresearch/detectron2>`_ for downstream
tasks.

COCO
^^^^
.. code-block::

    datasets/coco/
        annotations/
            captions_{train,val}2017.json
            instances_{train,val}2017.json
        train2017/
            # images in train2017 split
        val2017/
            # images in val2017 split

LVIS
^^^^
.. code-block::

    datasets/coco/
        train2017/
        val2017/
    datasets/lvis/
        lvis_v1.0_{train,val}.json

PASCAL VOC
^^^^^^^^^^
.. code-block::

    datasets/VOC2007/
        Annotations/
        ImageSets/
            Main/
                trainval.txt
                test.txt
        JPEGImages/

    datasets/VOC2012/
        # Same as VOC2007 above

ImageNet
^^^^^^^^
.. code-block::

    datasets/imagenet/
        train/
            # One directory per category with images in it
        val/
            # One directory per category with images in it
        ILSVRC2012_devkit_t12.tar.gz

iNaturalist 2018
^^^^^^^^^^^^^^^^
.. code-block::

    datasets/inaturalist/
        train_val2018/
        annotations/
            train2018.json
            val2018.json


ARCH 2021
^^^^^^^^^^^^^^^^
.. code-block::

    datasets/ARCH/
        annotations/
            captions_{all, train, val}.json
        books_set/
            images/
            captions.json
            README.md
        pubmed_set/
            images/
            captions.json
            README.md

-------------------------------------------------------------------------------


Build vocabulary
----------------

**Build a vocabulary out of COCO Captions ``train2017`` split.**

    .. code-block:: shell

        python scripts/build_vocabulary.py \
            --captions datasets/coco/annotations/captions_train2017.json \
            --vocab-size 10000 \
            --output-prefix datasets/vocab/coco_10k \
            --do-lower-case


**Build a vocabulary out of ARCH Captions ``train`` split.**

    .. code-block:: shell

        python scripts/build_vocabulary_arch.py \
            --captions datasets/ARCH/annotations/captions_train.json \
            --vocab-size 10000 \
            --output-prefix datasets/vocab/arch_10k \
            --do-lower-case

That's it! You are all set to use this codebase.
