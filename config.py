"""
Pipeline configuration

Every path has a sensible default relative to the repository root and can be
overridden through an environment variable, so the pipeline runs elsewhere
without editing any code:

    export ROOTSEG_DATA_DIR=/path/to/my/data
    python sam2_pipeline.py prepare

Command-line arguments take precedence over everything else
(--tiles_dir, --masks_dir, --prepared_data_dir, --checkpoint).

Expected layout for the raw data:

    <DATA_DIR>/
        tiles/      1024x1024 RGB images (.png or .jpg)
        masks/      matching masks, same file names, .png
"""

import os


def _env(name, default):
    """Return the environment variable if set, the default otherwise."""
    return os.environ.get(name, default)


# ----repository root--------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# SAM 2 package (build_sam, SAM2ImagePredictor). See the README for setup:
# git clone https://github.com/facebookresearch/sam2
SAM2_DIR = _env("SAM2_DIR", os.path.join(SCRIPT_DIR, "sam2"))

# ----data--------------------------------------------------------------------

DATA_DIR = _env("ROOTSEG_DATA_DIR", os.path.join(SCRIPT_DIR, "data"))
TILES_DIR = _env("ROOTSEG_TILES_DIR", os.path.join(DATA_DIR, "tiles"))
MASKS_DIR = _env("ROOTSEG_MASKS_DIR", os.path.join(DATA_DIR, "masks"))

# data converted to the layout expected by the SAM 2 video training framework:
# each image becomes a single-frame video sequence.

PREPARED_DATA_DIR = _env(
    "ROOTSEG_PREPARED_DIR", os.path.join(SCRIPT_DIR, "prepared_data")
)

# ----model-------------------------------------------------------------

MODEL_SIZE = _env("ROOTSEG_MODEL_SIZE", "tiny")  # tiny | small | base_plus | large

# hydra configuration used for fine-tuning
CONFIG_PATH = _env(
    "ROOTSEG_CONFIG_PATH",
    os.path.join(
        SAM2_DIR,
        "sam2",
        "configs",
        "sam2.1_training",
        "sam2.1_hiera_t_custom_soil.yaml",
    ),
)

# Inference configuration, relative to the sam2 package
INFERENCE_CONFIG = _env(
    "ROOTSEG_INFERENCE_CONFIG", "configs/sam2.1/sam2.1_hiera_t.yaml"
)

# Starting point: the pre-trained checkpoint released by Meta
PRETRAINED_CHECKPOINT = _env(
    "ROOTSEG_PRETRAINED_CHECKPOINT",
    os.path.join(SAM2_DIR, "checkpoints", f"sam2.1_hiera_{MODEL_SIZE}.pt"),
)

# ----outputs--------------------------------------------------------------------

LOG_DIR = _env("ROOTSEG_LOG_DIR", os.path.join(SCRIPT_DIR, "logs"))

FINETUNED_CHECKPOINT = _env(
    "ROOTSEG_FINETUNED_CHECKPOINT",
    os.path.join(LOG_DIR, "checkpoints", "checkpoint.pt"),
)

# ----data split------------------------------------------------------------------------

# 70/15/15 with a fixed seed for reproducibility.
# Note: this reimplements sklearn's train_test_split without sklearn, so the
# split differs from train_test_split(random_state=42) - the U-Net++ branch of the project
# used sklearn, so the two models were evaluated on different images. See the README.
SPLIT_SEED = int(_env("ROOTSEG_SPLIT_SEED", "42"))
TEST_FRAC = 0.3  # held out, then cut in half into val and test
VAL_OF_TEMP = 0.5

# ----runtime----------------------------------------------------------------------


def _default_device():
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


DEVICE = _env("ROOTSEG_DEVICE", _default_device())


def describe():
    """Print the effective configuration - handy to check a run"""
    print("Configuration")
    for name in (
        "SAM2_DIR",
        "DATA_DIR",
        "TILES_DIR",
        "MASKS_DIR",
        "PREPARED_DATA_DIR",
        "MODEL_SIZE",
        "PRETRAINED_CHECKPOINT",
        "FINETUNED_CHECKPOINT",
        "LOG_DIR",
        "DEVICE",
    ):
        print(f"{name:<22} {globals()[name]}")


if __name__ == "__main__":
    describe()
