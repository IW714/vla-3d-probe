"""
download_data.py  (sim env)

Writes the LIBERO config (so LIBERO never prompts interactively) and pulls the
LIBERO-Spatial demo files from Hugging Face into data/libero/libero_spatial/.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset import ensure_libero_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--download-dir", default="data/libero")
ap.add_argument("--datasets", default="libero_spatial")
args = ap.parse_args()

ensure_libero_config(args.download_dir)
from libero.libero.utils.download_utils import libero_dataset_download  # noqa: E402

libero_dataset_download(datasets=args.datasets, download_dir=args.download_dir, use_huggingface=True)
print(f"done: {os.path.join(args.download_dir, args.datasets)}")
