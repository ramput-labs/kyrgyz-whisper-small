#!/usr/bin/env python
"""Stream N Kyrgyz clips from Google FLEURS and write a TSV manifest."""

from __future__ import annotations

import argparse
import csv
import io
import tarfile
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/ky_kg"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    ap.add_argument("-n", "--num", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("data/fleurs_ky"))
    args = ap.parse_args()

    wav_dir = args.out / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    for old in wav_dir.glob("*.wav"):
        old.unlink()

    tsv = urllib.request.urlopen(f"{BASE}/{args.split}.tsv").read().decode("utf-8")
    refs = {}
    for row in csv.reader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        if len(row) >= 3:
            refs[row[1]] = row[2]

    rows: list[tuple[str, str]] = []
    with urllib.request.urlopen(f"{BASE}/audio/{args.split}.tar.gz") as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                name = Path(member.name).name
                if not member.isfile() or name not in refs:
                    continue
                data = tar.extractfile(member).read()
                (wav_dir / name).write_bytes(data)
                rows.append((f"wavs/{name}", refs[name]))
                print(f"[{len(rows):>3}/{args.num}] {name}  {refs[name][:70]}")
                if len(rows) >= args.num:
                    break

    manifest = args.out / "manifest.tsv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["path", "text"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} clips -> {manifest}")


if __name__ == "__main__":
    main()
