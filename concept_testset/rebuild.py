"""Rebuild the held-out concept test set from a local copy of ImageNet (ILSVRC2012).

Usage: python rebuild.py --imagenet_dir /path/to/imagenet [--out_dir img_concepts_outs]
"""
import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(imagenet_dir: str, out_dir: str, manifest: str):
    imagenet_dir, out_dir = Path(imagenet_dir), Path(out_dir)
    with open(manifest, newline='') as f:
        rows = list(csv.DictReader(f))

    missing = [r['imagenet_path'] for r in rows if not (imagenet_dir / r['imagenet_path']).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest images not found under {imagenet_dir}, e.g. {missing[:3]}. "
                                f"Expected the standard layout train/<wnid>/*.JPEG and val/<wnid>/*.JPEG.")

    for r in rows:
        src = imagenet_dir / r['imagenet_path']
        dst = out_dir / r['concept'] / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    counts = Counter(r['concept'] for r in rows)
    print(f"Copied {len(rows)} images for {len(counts)} concepts into {out_dir}")
    for concept, n in sorted(counts.items()):
        print(f"  {concept}: {n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet_dir", required=True, help="ImageNet root containing train/ and val/")
    parser.add_argument("--out_dir", default=str(HERE / 'img_concepts_outs'),
                        help="Output directory (default: concept_testset/img_concepts_outs)")
    parser.add_argument("--manifest", default=str(HERE / 'manifest.csv'))
    args = parser.parse_args()
    main(**vars(args))
