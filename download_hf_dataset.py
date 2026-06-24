#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import DownloadConfig, load_dataset


def resolve_cache_dir(cache_dir_arg: str | None) -> Path:
    if cache_dir_arg:
        return Path(cache_dir_arg).expanduser().resolve()
    env_cache = os.getenv("HF_DATASETS_CACHE")
    if env_cache:
        return Path(env_cache).expanduser().resolve()
    return Path.home() / ".cache" / "huggingface" / "datasets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face dataset into the selected local cache.",
    )
    parser.add_argument(
        "--dataset", type=str,
        help="Hugging Face dataset name (e.g. mnist, cifar10, zh-plus/tiny-imagenet, AI-Lab-Makerere/beans)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Dataset config/subset name (if required)",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Single split to download (e.g. train). If omitted, downloads all splits.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory to use. Priority: --cache-dir > HF_DATASETS_CACHE > HF default.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Dataset revision/tag/commit on Hugging Face.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token (required for private datasets).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_dir = resolve_cache_dir(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Keep behavior consistent with the rest of the project reading HF_DATASETS_CACHE.
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir)

    print(f"[info] Dataset: {args.dataset}")
    if args.config:
        print(f"[info] Config: {args.config}")
    if args.split:
        print(f"[info] Split: {args.split}")
    if args.revision:
        print(f"[info] Revision: {args.revision}")
    print(f"[info] Cache dir: {cache_dir}")

    download_config = DownloadConfig(resume_download=True)

    ds = load_dataset(
        path=args.dataset,
        name=args.config,
        split=args.split,
        cache_dir=str(cache_dir),
        revision=args.revision,
        token=args.token,
        download_config=download_config,
    )

    if args.split:
        print(f"[ok] Downloaded split: {args.split}")
        print(f"[ok] Number of samples: {len(ds)}")
    else:
        splits = list(ds.keys())
        print(f"[ok] Downloaded splits: {splits}")
        for split_name in splits:
            print(f"[ok] {split_name}: {len(ds[split_name])} samples")

    print("[ok] Download completed in local cache.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
