"""Build the ROI sub-window cache for one (delta_t, roi_k) pair.

roigesture_conv1d and roigesture_channels share the same cache
(./cache/DVS_ROI_<fs>_fwdPass_trte<dt>k<k>_tete<dt>k<k>_2/), and so do all seeds,
so it is built once per pair BEFORE the job array: concurrent jobs writing the
same tonic DiskCachedDataset files could corrupt them (unreadable samples are
then silently skipped by dataset_to_numpy).

Usage:
    python build_roi_k_cache.py --delta_t 10000 --roi_k 10
"""
import argparse
from pathlib import Path

from exp_config import create_config_file, set_active_config, load_cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delta_t", type=int, required=True, help="frame length in us")
    ap.add_argument("--roi_k", type=int, required=True, help="sub-windows per frame")
    ap.add_argument("--frame_size", type=int, default=32)
    args = ap.parse_args()

    cfg_dir = Path("cache_cfg") / f"roi_k_dt{args.delta_t}_k{args.roi_k}"
    cfg_path = create_config_file(cfg_dir, {
        "dataset": "roigesture_conv1d", "mode": "fwdPass", "name": str(cfg_dir),
        "delta_t": args.delta_t, "roi_k": args.roi_k,
        "train_framing": "delta_t", "test_framing": "delta_t",
    })
    set_active_config(cfg_path)
    cfg = load_cfg(force=True)

    from components.gesture_dataset import get_ROI_numpy
    (x_train, _), (x_test, _) = get_ROI_numpy(cfg, frame_size=args.frame_size)
    print(f"[build_roi_k_cache] delta_t={args.delta_t} roi_k={args.roi_k}: "
          f"{len(x_train)} train / {len(x_test)} test recordings kept")


if __name__ == "__main__":
    main()
