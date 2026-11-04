"""The images-on-disk RDP datasets (rdp_shims/peel_lazy_datasets.py) must give exactly the samples and the normalizer
of RDP's own in-memory datasets (their normalizer is fitted without reading images). Uses the small test dataset and
saved run configs; skipped when they are not present."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ZARR = ROOT / "data" / "rdp" / "test_stroke_zarr"
CASES = [  # saved run config, images-on-disk class (DP dataset; RDP stage-2 dataset, which also needs the tokenizer)
    ("test_dp", "peel_lazy_datasets.RealImageTactileDatasetImagesOnDisk"),
    ("test_ldp", "peel_lazy_datasets.RealImageTactileLatentDiffusionDatasetImagesOnDisk"),
]


def _flat(sample, prefix=""):
    out = {}
    for k, v in sample.items():
        if isinstance(v, dict):
            out.update(_flat(v, prefix + k + "/"))
        else:
            out[prefix + k] = v.numpy()
    return out


@pytest.mark.parametrize("run,target", CASES)
def test_images_on_disk_dataset_matches_in_memory(run, target):
    cfg_path = ROOT / "runs" / run / ".hydra" / "config.yaml"
    if not (cfg_path.exists() and ZARR.exists()):
        pytest.skip(f"needs runs/{run} and test_stroke_zarr")
    for p in (str(ROOT / "rdp_shims"), str(ROOT / "third_party" / "rdp")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import hydra
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)  # as RDP's train.py does
    cfg = OmegaConf.load(cfg_path)
    cfg.task.dataset.dataset_path = str(ZARR)
    ref = hydra.utils.instantiate(cfg.task.dataset)
    cfg.task.dataset._target_ = target
    lazy = hydra.utils.instantiate(cfg.task.dataset)
    assert len(ref) == len(lazy) > 0
    assert not isinstance(lazy.replay_buffer["external_img"], np.ndarray)  # images really stay on disk
    for i in np.linspace(0, len(ref) - 1, 25).astype(int):
        a, b = _flat(ref[i]), _flat(lazy[i])
        assert a.keys() == b.keys()
        for k in a:
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)
    import torch

    torch.manual_seed(0)  # the stage-2 latent-action fit samples from the tokenizer's posterior
    sa = ref.get_normalizer().state_dict()
    torch.manual_seed(0)
    sb = lazy.get_normalizer().state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        np.testing.assert_array_equal(sa[k].numpy(), sb[k].numpy(), err_msg=k)
