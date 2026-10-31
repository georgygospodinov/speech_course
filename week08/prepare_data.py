"""Download the course's VoxTube subset and verify the original archives."""

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

PUBLIC_LINKS = {
    "train": "https://disk.yandex.ru/d/HBEZ7MbCepdgzw",
    "dev": "https://disk.yandex.ru/d/LKDitPbiU5XS2A",
}


def download_split(split, data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    info = json.loads(subprocess.check_output([
        "curl", "-fLsS", "--retry", "3", "--get",
        "https://cloud-api.yandex.net/v1/disk/public/resources",
        "--data-urlencode", f"public_key={PUBLIC_LINKS[split]}",
    ]))
    archive = data_dir / info["name"]
    if not archive.exists() or archive.stat().st_size != info["size"]:
        partial = archive.with_suffix(".tar.part")
        print(f"Downloading {split}: {info['size'] / 1e6:.0f} MB", flush=True)
        subprocess.run([
            "curl", "-fL", "--retry", "3", "--continue-at", "-",
            "--output", str(partial), info["file"],
        ], check=True)
        partial.rename(archive)
    digest = hashlib.sha256()
    with archive.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != info["sha256"]:
        raise ValueError(f"SHA256 mismatch: {archive}; remove it and download again")
    print(f"{split}: SHA256 OK; extracting to {data_dir.resolve()}", flush=True)
    with tarfile.open(archive) as source:
        source.extractall(data_dir, filter="data")
    print(f"{split}: ready", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", nargs="+", choices=PUBLIC_LINKS, default=list(PUBLIC_LINKS))
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    for split in args.split:
        download_split(split, args.data_dir)
