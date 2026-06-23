"""
Balance an image dataset across classes, then split into train/test.

What this script does, in order:
1. Counts images in every class subfolder inside TRAIN_DIR.
2. Finds the smallest class count and reduces every other class down to
   that number. By default the "extra" images are MOVED to a backup
   folder (nothing is deleted) -- pass --delete-excess if you'd rather
   delete them outright.
3. Once all classes are equal size, moves a random 20% (configurable via
   --test-split) of each class from TRAIN_DIR/<class> to TEST_DIR/<class>,
   leaving 80% behind in TRAIN_DIR/<class>.

Run with --dry-run first to preview exactly what would happen without
touching any files.

Assumptions:
- TRAIN_DIR and TEST_DIR each contain subfolders with matching names
  (e.g. train/cats, train/dogs <-> test/cats, test/dogs).
- Images sit directly inside each class subfolder (no further nesting).

Usage:
    python balance_and_split.py --train train --test test --dry-run
    python balance_and_split.py --train train --test test
    python balance_and_split.py --train train --test test --delete-excess --test-split 0.15
"""

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def list_images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def class_dirs(folder: Path):
    return sorted(d for d in folder.iterdir() if d.is_dir())


def balance_classes(train_dir, backup_dir, delete_excess, seed, dry_run):
    dirs = class_dirs(train_dir)
    counts = {d.name: len(list_images(d)) for d in dirs}
    if not counts:
        raise SystemExit(f"No class subfolders found in {train_dir}")

    min_count = min(counts.values())
    print("Current class counts:", counts)
    print(f"--> Balancing all classes down to {min_count} images each\n")

    rng = random.Random(seed)
    for d in dirs:
        images = list_images(d)
        excess = len(images) - min_count
        if excess <= 0:
            continue
        rng.shuffle(images)
        to_remove = images[:excess]

        if delete_excess:
            action = f"DELETE {len(to_remove)} images from '{d.name}'"
            if not dry_run:
                for img in to_remove:
                    img.unlink()
        else:
            target = backup_dir / d.name
            action = f"MOVE {len(to_remove)} excess images from '{d.name}' to {target}"
            if not dry_run:
                target.mkdir(parents=True, exist_ok=True)
                for img in to_remove:
                    shutil.move(str(img), str(target / img.name))
        print(("[DRY RUN] " if dry_run else "") + action)

    return min_count


def split_train_test(train_dir, test_dir, test_split, seed, dry_run):
    rng = random.Random(seed + 1)
    dirs = class_dirs(train_dir)
    print(f"\n--> Moving {test_split:.0%} of each balanced class to test\n")

    for d in dirs:
        images = list_images(d)
        rng.shuffle(images)
        n_test = round(len(images) * test_split)
        to_move = images[:n_test]
        target = test_dir / d.name

        msg = f"{d.name}: move {len(to_move)} to test, keep {len(images) - len(to_move)} in train"
        print(("[DRY RUN] " if dry_run else "") + msg)

        if not dry_run:
            target.mkdir(parents=True, exist_ok=True)
            for img in to_move:
                shutil.move(str(img), str(target / img.name))


def main():
    p = argparse.ArgumentParser(description="Balance train classes, then 80/20 split into train/test.")
    p.add_argument("--train", default="train", help="Path to train folder")
    p.add_argument("--test", default="test", help="Path to test folder")
    p.add_argument("--backup", default="train_removed", help="Where excess images go when balancing (if not deleting)")
    p.add_argument("--test-split", type=float, default=0.2, help="Fraction moved to test, e.g. 0.2 = 20%%")
    p.add_argument("--seed", type=int, default=42, help="Random seed, for reproducible splits")
    p.add_argument("--delete-excess", action="store_true", help="Delete excess images instead of backing them up")
    p.add_argument("--dry-run", action="store_true", help="Preview actions without moving/deleting anything")
    args = p.parse_args()

    train_dir, test_dir, backup_dir = Path(args.train), Path(args.test), Path(args.backup)
    if not train_dir.is_dir():
        raise SystemExit(f"Train folder not found: {train_dir}")
    if not test_dir.is_dir():
        raise SystemExit(f"Test folder not found: {test_dir}")

    balance_classes(train_dir, backup_dir, args.delete_excess, args.seed, args.dry_run)
    split_train_test(train_dir, test_dir, args.test_split, args.seed, args.dry_run)

    print("\nDone." + (" (dry run -- no files were changed)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
