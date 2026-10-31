"""Cache YouTube source-video thumbnails for offline pair browsing."""

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


DATA = Path(__file__).parent / "data"


def download_thumbnail(video_id):
    target = DATA / "thumbnails" / f"{video_id}.jpg"
    if target.exists():
        return True
    result = subprocess.run([
        "curl", "-fLsS", "--retry", "2", "--max-time", "15",
        "--output", str(target), f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
    ], capture_output=True)
    return result.returncode == 0


if __name__ == "__main__":
    (DATA / "thumbnails").mkdir(parents=True, exist_ok=True)
    videos = sorted({p.parent.name for p in (DATA / "dev").rglob("*.wav")})
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(download_thumbnail, videos))
    print(f"Cached {sum(results)}/{len(videos)} source thumbnails; videos need internet.")
