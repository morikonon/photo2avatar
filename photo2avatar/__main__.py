from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .pipeline import run, run_mean_body


def main() -> None:
    parser = argparse.ArgumentParser(prog="photo2avatar", description="Photo -> rigged, animated 3D avatar (GLB)")
    parser.add_argument("image", nargs="?", type=Path, help="full-body photo of one person")
    parser.add_argument("-o", "--out", type=Path, default=Path("out"))
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], help="default: best available")
    parser.add_argument("--passes", type=int, default=2, help="HMR2 passes, 1 = whole image only")
    parser.add_argument("--box", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                        help="person box for the first pass")
    parser.add_argument("--mean-body", action="store_true", help="skip the photo and export the average body")
    args = parser.parse_args()

    if args.mean_body:
        report = run_mean_body(args.out)
    elif args.image is None:
        parser.error("pass an image path or --mean-body")
    else:
        box = None if args.box is None else np.array(args.box)
        report = run(args.image, args.out, device=args.device, passes=args.passes, box=box)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
