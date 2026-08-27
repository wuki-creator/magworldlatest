from __future__ import annotations

import concurrent.futures
import os
import pathlib
import subprocess
import time

URL = (
    "https://huggingface.co/datasets/arcinstitute/Replogle-Nadig-Preprint/"
    "resolve/main/K562_essential_normalized_singlecell_01.h5ad?download=true"
)
SIZE = 10_662_212_844
CHUNK = 64 * 1024 * 1024
OUT = pathlib.Path("/root/magworld/data/replogle_k562_essential/K562_essential_normalized_singlecell_01.h5ad")
PARTS = pathlib.Path(str(OUT) + ".parts")


def fetch(item: tuple[int, int, int]) -> int:
    index, start, end = item
    path = PARTS / f"{index:04d}"
    expected = end - start + 1
    for attempt in range(10):
        if path.exists() and path.stat().st_size == expected:
            return index
        path.unlink(missing_ok=True)
        result = subprocess.run(
            [
                "curl", "-L", "--fail", "--retry", "5", "--retry-delay", "2",
                "--connect-timeout", "30", "-sS", "-r", f"{start}-{end}",
                URL, "-o", str(path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0 and path.exists() and path.stat().st_size == expected:
            return index
        time.sleep(min(60, 2**attempt))
    raise RuntimeError(f"part {index} failed: expected {expected} bytes")


def main() -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    ranges = [
        (i, i * CHUNK, min(SIZE - 1, (i + 1) * CHUNK - 1))
        for i in range((SIZE + CHUNK - 1) // CHUNK)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        for done, index in enumerate(pool.map(fetch, ranges), 1):
            if done % 10 == 0 or done == len(ranges):
                print(f"parts={done}/{len(ranges)}", flush=True)
    assembling = OUT.with_name(OUT.name + ".assembling")
    with assembling.open("wb") as destination:
        for index, _, _ in ranges:
            with (PARTS / f"{index:04d}").open("rb") as source:
                while block := source.read(16 * 1024 * 1024):
                    destination.write(block)
    if assembling.stat().st_size != SIZE:
        raise RuntimeError("assembled file size mismatch")
    os.replace(assembling, OUT)
    for part in PARTS.iterdir():
        part.unlink()
    PARTS.rmdir()
    print(f"DOWNLOAD_COMPLETE {OUT} {OUT.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
