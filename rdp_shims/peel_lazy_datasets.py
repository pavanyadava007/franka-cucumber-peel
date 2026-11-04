"""RDP dataset classes that keep camera images on disk, for datasets larger than RAM.

RDP's ReplayBuffer.copy_from_path decompresses every array into memory. The v1 + v2 stroke dataset has 100k frames
from two cameras (7.4 GB of images), more than the 12 GB WSL VM holds next to training, so these subclasses keep the
image arrays (N, H, W, C) as read-only zarr arrays on disk (one chunk per frame, read when a sample needs it) and load
everything else into memory exactly as before. The samples and the normalizer are identical
(tests/test_lazy_dataset.py); only memory use and speed change. The official RDP code is not modified: the task
configs point `_target_` here.
"""

import os
from contextlib import contextmanager

import numpy as np
import zarr
from reactive_diffusion_policy.common.normalize_util import get_image_range_normalizer
from reactive_diffusion_policy.common.replay_buffer import ReplayBuffer
from reactive_diffusion_policy.dataset.real_image_tactile_dataset import RealImageTactileDataset
from reactive_diffusion_policy.dataset.real_image_tactile_latent_diffusion_dataset import (
    RealImageTactileLatentDiffusionDataset,
)


def _copy_from_path_images_on_disk(cls, zarr_path, keys=None, **kwargs):
    src = zarr.open(os.path.expanduser(zarr_path), mode="r")
    meta = {k: (np.array(v) if len(v.shape) == 0 else v[:]) for k, v in src["meta"].items()}
    data = {}
    for k in keys if keys is not None else src["data"].keys():
        arr = src["data"][k]
        data[k] = arr if arr.ndim == 4 else arr[:]
    return cls(root={"meta": meta, "data": data})


@contextmanager
def images_on_disk():
    """While active, ReplayBuffer.copy_from_path leaves image arrays on disk."""
    orig = ReplayBuffer.__dict__["copy_from_path"]
    ReplayBuffer.copy_from_path = classmethod(_copy_from_path_images_on_disk)
    try:
        yield
    finally:
        ReplayBuffer.copy_from_path = orig


def _normalizer_without_images(ds, fit):
    """RDP fits the normalizer by reading every sample, images included (relative actions, and the latent actions of
    stage 2), although only poses, wrenches and actions are used; images get a fixed [0, 1] range. Reading 100k frames
    from disk takes over an hour, so `fit` runs with the image keys hidden and the image normalizers are added as RDP
    adds them."""
    image_keys = set(ds.rgb_keys + ds.extended_rgb_keys)
    saved = ds.sampler.keys, ds.rgb_keys, ds.extended_rgb_keys
    ds.sampler.keys = [k for k in ds.sampler.keys if k not in image_keys]
    ds.rgb_keys, ds.extended_rgb_keys = [], []
    try:
        normalizer = fit()
    finally:
        ds.sampler.keys, ds.rgb_keys, ds.extended_rgb_keys = saved
    for key in image_keys:
        normalizer[key] = get_image_range_normalizer()
    return normalizer


class RealImageTactileDatasetImagesOnDisk(RealImageTactileDataset):
    def __init__(self, *args, **kwargs):
        with images_on_disk():
            super().__init__(*args, **kwargs)

    def get_normalizer(self, **kwargs):
        fit = super().get_normalizer
        return _normalizer_without_images(self, lambda: fit(**kwargs))


class RealImageTactileLatentDiffusionDatasetImagesOnDisk(RealImageTactileLatentDiffusionDataset):
    def __init__(self, *args, **kwargs):
        with images_on_disk():
            super().__init__(*args, **kwargs)

    def get_normalizer(self, **kwargs):
        fit = super().get_normalizer
        return _normalizer_without_images(self, lambda: fit(**kwargs))
