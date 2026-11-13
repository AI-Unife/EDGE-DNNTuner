"""Build the ROI sub-window cache for one (delta_t, roi_k) pair.

roigesture_conv1d and roigesture_channels share the same cache
(./cache/DVS_ROI_<fs>_fwdPass_trte<dt>k<k>_tete<dt>k<k>_2/), and so do all seeds,
so it is built once per pair BEFORE the job array: concurrent jobs writing the
same tonic DiskCachedDataset files could corrupt them (unreadable samples are
then silently skipped by dataset_to_numpy).
On success it writes <cache folder>/.complete, so slurm/submit_roi_k.sh does
not rebuild that pair again.

--roi_k 0 builds instead the plain delta_t ROI cache
(./cache/DVS_ROI_<fs>_fwdPass_trte<dt>_tete<dt>_2/), shared by roigesture_coords,
roigesture_matrix and roigesture_matrix_fc (slurm/submit_roi_all.sh).

Usage:
    python build_roi_k_cache.py --delta_t 10000 --roi_k 10
    python build_roi_k_cache.py --delta_t 10000 --roi_k 0
"""
import argparse
from pathlib import Path

from exp_config import create_config_file, set_active_config, load_cfg

CACHE_MARKER = ".complete"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delta_t", type=int, required=True, help="frame length in us")
    ap.add_argument("--roi_k", type=int, required=True, help="sub-windows per frame (0: plain delta_t ROI cache)")
    ap.add_argument("--frame_size", type=int, default=32)
    args = ap.parse_args()

    cfg_dir = Path("cache_cfg") / f"roi_k_dt{args.delta_t}_k{args.roi_k}"
    cfg_dict = {
        "dataset": "roigesture_conv1d" if args.roi_k else "roigesture_matrix",
        "mode": "fwdPass", "name": str(cfg_dir), "delta_t": args.delta_t,
        "train_framing": "delta_t", "test_framing": "delta_t",
    }
    if args.roi_k:
        cfg_dict["roi_k"] = args.roi_k
    cfg_path = create_config_file(cfg_dir, cfg_dict)
    set_active_config(cfg_path)
    cfg = load_cfg(force=True)

    from components.gesture_dataset import get_ROI_numpy
    (x_train, _), (x_test, _) = get_ROI_numpy(cfg, frame_size=args.frame_size)
    print(f"[build_roi_k_cache] delta_t={args.delta_t} roi_k={args.roi_k}: "
          f"{len(x_train)} train / {len(x_test)} test recordings kept")
    if not len(x_train) or not len(x_test):
        raise SystemExit("[build_roi_k_cache] empty split: ROI dataset missing or unreadable, cache NOT marked complete")

    # Marker read by slurm/submit_roi_k.sh to skip this pair next time
    # (same folder name as get_ROI_numpy's cache_dir).
    tag = f"te{args.delta_t}k{args.roi_k}" if args.roi_k else f"te{args.delta_t}"
    cache_dir = Path("cache") / f"DVS_ROI_{args.frame_size}_fwdPass_tr{tag}_te{tag}_{cfg.channels}"
    if not cache_dir.is_dir():
        raise SystemExit(f"[build_roi_k_cache] expected cache folder {cache_dir} not found")
    (cache_dir / CACHE_MARKER).write_text(f"{len(x_train)} train / {len(x_test)} test\n")
    print(f"[build_roi_k_cache] marked {cache_dir} complete")


if __name__ == "__main__":
    main()
