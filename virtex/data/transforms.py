import random
from typing import List
import unicodedata

import torchvision
import albumentations as alb
import cv2

from virtex.data.tokenizers import SentencePieceBPETokenizer


class CaptionOnlyTransform(alb.BasicTransform):
    r"""
    Base class for custom `albumentations <https://albumentations.readthedocs.io/en/latest/>`_
    transform, which can transform captions. Captions may be ``str``, or tokens
    (``List[int]``) as per implementation of :meth:`apply_to_caption`. These
    transforms will have consistent API as other transforms from albumentations.
    """

    @property
    def targets(self):
        return {"caption": self.apply_to_caption}

    def apply_to_caption(self, caption, **params):
        raise NotImplementedError

    def update_params(self, params, **kwargs):
        # Super class adds "width" and "height" but we don't have image here.
        return params


class ImageCaptionTransform(alb.BasicTransform):
    r"""
    Similar to :class:`~virtex.data.transforms.CaptionOnlyTransform`, this
    extends super class to work on ``(image, caption)`` pair together.
    """

    @property
    def targets(self):
        return {"image": self.apply, "caption": self.apply_to_caption}

    def apply_to_caption(self):
        raise NotImplementedError


class NormalizeCaption(CaptionOnlyTransform):
    r"""
    Perform common normalization with caption: lowercase, trim leading and
    trailing whitespaces, NFKD normalization and strip accents.

    Examples:
        >>> normalize = NormalizeCaption(always_apply=True)
        >>> out = normalize(caption="Some caption input here.")  # keys: {"caption"}
    """

    def __init__(self):
        # `always_apply = True` because this is essential part of pipeline.
        super().__init__(always_apply=True)

    def apply_to_caption(self, caption: str, **params) -> str:
        caption = caption.lower()
        caption = unicodedata.normalize("NFKD", caption)
        caption = "".join([chr for chr in caption if not unicodedata.combining(chr)])
        return caption


class TokenizeCaption(CaptionOnlyTransform):
    r"""
    Tokenize a caption (``str``) to list of tokens (``List[int]``) by the
    mapping defined in :attr:`tokenizer`.

    Args:
        tokenizer: A :class:`~virtex.data.tokenizers.SentencePieceBPETokenizer`
            which encodes a caption into tokens.
        add_boundaries: Whether to add ``[SOS]``/``[EOS]`` tokens from tokenizer.

    Examples:
        >>> tokenizer = SentencePieceBPETokenizer("coco.vocab", "coco.model")
        >>> tokenize = TokenizeCaption(tokenizer, always_apply=True)
        >>> out = tokenize(caption="Some caption input here.")  # keys: {"caption"}
    """

    def __init__(self, tokenizer: SentencePieceBPETokenizer):
        # `always_apply = True` because this is essential part of pipeline.
        super().__init__(always_apply=True)
        self.tokenizer = tokenizer

    def apply_to_caption(self, caption: str, **params) -> List[int]:
        token_indices: List[int] = self.tokenizer.encode(caption)

        # Add boundary tokens.
        token_indices.insert(0, self.tokenizer.token_to_id("[SOS]"))
        token_indices.append(self.tokenizer.token_to_id("[EOS]"))
        return token_indices

    def get_transform_init_args_names(self):
        return ("tokenizer",)


class TruncateCaptionTokens(CaptionOnlyTransform):
    r"""
    Truncate a list of caption tokens (``List[int]``) to maximum length.

    Args:
        max_caption_length: Maximum number of tokens to keep in output caption
            tokens. Extra tokens will be trimmed from the right end of token list.

    Examples:
        >>> truncate = TruncateCaptionTokens(max_caption_length=5)
        >>> out = truncate(caption=[2, 35, 41, 67, 98, 50, 3])
        >>> out["caption"]
        [2, 35, 41, 67, 98]
    """

    def __init__(self, max_caption_length: int):
        # `always_apply = True` because this is essential part of pipeline.
        super().__init__(always_apply=True)
        self.max_caption_length = max_caption_length

    def apply_to_caption(self, caption: List[int], **params) -> List[int]:
        return caption[: self.max_caption_length]

    def get_transform_init_args_names(self):
        return ("max_caption_length",)


class HorizontalFlip(ImageCaptionTransform):
    r"""
    Flip the image horizontally randomly (equally likely) and replace the
    word "left" with "right" in the caption.

    .. note::

        This transform can also work on images only (without the captions).
        Its behavior will be same as albumentations
        :class:`~albumentations.augmentations.transforms.HorizontalFlip`.

    Examples:
        >>> flip = HorizontalFlip(p=0.5)
        >>> out1 = flip(image=image, caption=caption)  # keys: {"image", "caption"}
        >>> # Also works with images (without caption).
        >>> out2 = flip(image=image)  # keys: {"image"}

    """

    def apply(self, img, **params):
        # 1 stands for flipping in y-axis, not for the probability
        return cv2.flip(img, 1)

    def apply_to_caption(self, caption, **params):
        caption = (
            caption.replace("left", "[TMP]")
            .replace("right", "left")
            .replace("[TMP]", "right")
        )
        return caption


class TensorHorizontalFlip(HorizontalFlip):
    r"""
    Flip the batch of images (tensor of shape: ..., H, W) horizontally
    randomly (equally likely) and replace the word "left" with "right" in the
    caption.

    .. note::

        This transform can also work on images only (without the captions).
        It only overrides the apply method for images so that it can flip a
        tensor of shape (B, C, H, W) and not just an image of shape (H, W, C)

    Examples:
        >>> tensor_flip = TensorHorizontalFlip(p=0.5)
        >>> out1 = tensor_flip(image=image_tensor, caption=caption)  # keys: {
        "image", "caption"}
        >>> # Also works with image-tensors (without caption).
        >>> out2 = tensor_flip(image=image_tensor)  # keys: {"image"}

    """

    def apply(self, img_tensor, **params):
        # flip the tensor (`p` can be specified because both HorizontalFlip and
        # TensorHorizontalFlip inherit from alb.BasicTransform which allows
        # to specify the probability of execution)
        return torchvision.transforms.functional.hflip(img_tensor)


class RandomResizedSquareCrop(alb.RandomResizedCrop):
    r"""
    A variant of :class:`albumentations.augmentations.transforms.RandomResizedCrop`
    which assumes a square crop (width = height). Everything else is same.

    Args:
        size: Dimension of the width and height of the cropped image.
    """

    def __init__(self, size: int, *args, **kwargs):
        super().__init__(height=size, width=size, *args, **kwargs)


class CenterSquareCrop(alb.CenterCrop):
    r"""
    A variant of :class:`albumentations.augmentations.transforms.CenterCrop`
    which assumes a square crop (width = height). Everything else is same.

    Args:
        size: Dimension of the width and height of the cropped image.
    """

    def __init__(self, size: int, *args, **kwargs):
        super().__init__(height=size, width=size, *args, **kwargs)


class SquareResize(alb.Resize):
    r"""
    A variant of :class:`albumentations.augmentations.transforms.Resize` which
    assumes a square resize (width = height). Everything else is same.

    Args:
        size: Dimension of the width and height of the cropped image.
    """

    def __init__(self, size: int, *args, **kwargs):
        super().__init__(height=size, width=size, *args, **kwargs)


# =============================================================================
#   SOME COMMON CONSTANTS AND IMAGE TRANSFORMS:
#   These serve as references here, and are used as default params in many
#   dataset class constructors.
# -----------------------------------------------------------------------------

IMAGENET_COLOR_MEAN = (0.485, 0.456, 0.406)
r"""ImageNet color normalization mean in RGB format (values in 0-1)."""

IMAGENET_COLOR_STD = (0.229, 0.224, 0.225)
r"""ImageNet color normalization std in RGB format (values in 0-1)."""

DEFAULT_IMAGE_TRANSFORM = alb.Compose(
    [
        alb.SmallestMaxSize(256, p=1.0),
        CenterSquareCrop(224, p=1.0),
        alb.Normalize(mean=IMAGENET_COLOR_MEAN, std=IMAGENET_COLOR_STD, p=1.0),
    ]
)
r"""Default transform without any data augmentation (during pretraining)."""

# -----------------------------------------------------------------------------
ARCH_DEFAULT_IMAGE_TRANSFORM = alb.Compose(
    [
        alb.SmallestMaxSize(256, p=1.0),
        CenterSquareCrop(224, p=1.0),
    ]
)
r"""Default transform for ARCH dataset without any data augmentation (during pretraining)."""

DEFAULT_FLIP_TRANSFORM = TensorHorizontalFlip(p=0.5)
# =============================================================================
