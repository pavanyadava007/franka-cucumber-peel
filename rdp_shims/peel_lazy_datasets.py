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


class RealImageTactileDatasetImagesOnDisk(RealImageTactileDataset):
    def __init__(self, *args, **kwargs):
        with images_on_disk():
            super().__init__(*args, **kwargs)

    def get_normalizer(self, **kwargs):
        # With relative actions RDP fits the normalizer by reading every sample, images included, although only the
        # poses and actions are used (images get a fixed [0, 1] range). Reading 100k frames from disk takes over an
        # hour, so the pass runs without the image keys; the image normalizers are added as RDP would add them.
        image_keys = set(self.rgb_keys + self.extended_rgb_keys)
        saved = self.sampler.keys, self.rgb_keys, self.extended_rgb_keys
        self.sampler.keys = [k for k in self.sampler.keys if k not in image_keys]
        self.rgb_keys, self.extended_rgb_keys = [], []
        try:
            normalizer = super().get_normalizer(**kwargs)
        finally:
            self.sampler.keys, self.rgb_keys, self.extended_rgb_keys = saved
        for key in image_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer


class RealImageTactileLatentDiffusionDatasetImagesOnDisk(RealImageTactileLatentDiffusionDataset):
    def __init__(self, *args, **kwargs):
        with images_on_disk():
            super().__init__(*args, **kwargs)
