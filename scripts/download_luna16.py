"""Download the official LUNA16 release from Zenodo with resume and checksum checks."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


ZENODO_RECORDS = {
    0: (2595813, "1065b0f42b8c25cf29260fd924a3c3a2", 6_811_924_508),
    1: (2595813, "2eda28bb123543074f5cd849499b87ff", 6_334_778_552),
    2: (2595813, "685b484b085bcdb81c525f3b1bf97c51", 7_257_937_108),
    3: (2595813, "54707e9e8954af29326eed60f38ea321", 6_896_620_114),
    4: (2595813, "98225ccb1a41fc434631f63f4284796a", 6_856_144_330),
    # The earlier Zenodo version (record 2595813) has a subset5 archive whose
    # checksum is valid but whose two RAW members fail decompression.  The
    # LUNA16 download page now resolves Part 1 to record 3723295, which contains
    # the corrected archive with the same byte size and a different checksum.
    5: (3723295, "395ec714c4123cf213fb7b08f05ee9cc", 6_610_460_097),
    6: (2595813, "d162df1444a6f2674ab8273c7f9e1520", 6_531_050_274),
    # Part 2 also received corrected archives after its original Zenodo
    # release.  Use the newest version reached by the official concept DOI.
    7: (4121926, "e99b8921990d1414bb5e92130371e8a3", 6_313_598_213),
    8: (4121926, "38260ca9a8741888997bcebedfabc3f1", 6_025_767_505),
    9: (4121926, "e55c473bbeebd712eb2224a58998be8e", 6_699_650_017),
}
ANNOTATIONS = (2595813, "f4404df491aee6445a4485061bfdffdd", 136_986)
CHUNK_SIZE = 8 * 1024 * 1024


def _zenodo_url(record_id, filename):
    return f"https://zenodo.org/api/records/{record_id}/files/{filename}/content"


def _md5(path):
    digest = hashlib.md5()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_once(url, destination, expected_size, expected_md5):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    if destination.exists():
        if destination.stat().st_size == expected_size and _md5(destination) == expected_md5:
            print(f"Verified existing file: {destination.name}")
            return destination
        raise RuntimeError(
            f"Existing file failed validation: {destination}. "
            "Move it aside before retrying; it will not be overwritten automatically."
        )

    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")

    with urllib.request.urlopen(request, timeout=120) as response:
        resumed = offset > 0 and response.status == 206
        if offset and not resumed:
            raise RuntimeError(
                f"The server did not accept resume for {destination.name}. "
                f"Partial file kept at {partial}; no data was overwritten."
            )

        mode = "ab" if resumed else "wb"
        downloaded = offset
        next_report = downloaded + 512 * 1024 * 1024
        with partial.open(mode) as file_obj:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                file_obj.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    print(f"{destination.name}: {downloaded / 1024**3:.1f} / {expected_size / 1024**3:.1f} GiB")
                    next_report += 512 * 1024 * 1024

    if partial.stat().st_size != expected_size:
        raise RuntimeError(
            f"Size mismatch for {destination.name}: got {partial.stat().st_size}, expected {expected_size}. "
            "Partial download was kept for inspection or resume."
        )
    if _md5(partial) != expected_md5:
        raise RuntimeError(f"Checksum mismatch for {destination.name}; partial file kept at {partial}.")

    os.replace(partial, destination)
    print(f"Downloaded and verified: {destination.name}")
    return destination


def _is_transient_download_error(exc):
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {408, 429, 500, 502, 503, 504}
    return isinstance(
        exc,
        (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
        ),
    )


def _download(url, destination, expected_size, expected_md5, max_attempts=8):
    for attempt in range(1, max_attempts + 1):
        try:
            return _download_once(url, destination, expected_size, expected_md5)
        except Exception as exc:
            if not _is_transient_download_error(exc) or attempt >= max_attempts:
                raise
            delay = min(60, 5 * 2 ** (attempt - 1))
            print(
                f"Transient download error for {Path(destination).name}: {exc}. "
                f"Retrying from the saved partial file in {delay} seconds "
                f"(attempt {attempt + 1}/{max_attempts}).",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError("unreachable")


def _extract_subset(archive, output_root, subset_index):
    output_root = Path(output_root)
    subset_dir = output_root / f"subset{subset_index}"
    if subset_dir.exists() and any(subset_dir.glob("*.mhd")):
        print(f"Already extracted: {subset_dir.name}")
        return
    if subset_dir.exists():
        raise RuntimeError(
            f"Extraction target exists but is incomplete: {subset_dir}. "
            "It will not be overwritten automatically."
        )

    staging = output_root / f".subset{subset_index}.extracting"
    if staging.exists():
        if not staging.is_dir():
            raise RuntimeError(f"Extraction staging path is not a directory: {staging}")
        print(f"Resuming extraction in existing staging directory: {staging.name}")
    else:
        staging.mkdir(parents=True)

    with zipfile.ZipFile(archive) as zip_file:
        zip_file.extractall(staging)

    extracted = staging / f"subset{subset_index}"
    if not extracted.exists():
        extracted = staging
    if not any(extracted.glob("*.mhd")):
        raise RuntimeError(f"Archive did not contain expected .mhd files: {archive}")

    if extracted == staging:
        staging.rename(subset_dir)
    else:
        shutil.move(str(extracted), str(subset_dir))
        staging.rmdir()
    print(f"Extracted: {subset_dir.name}")


def _parse_subsets(value):
    if value.lower() == "all":
        return list(range(10))
    subsets = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    invalid = [item for item in subsets if item not in ZENODO_RECORDS]
    if invalid or not subsets:
        raise argparse.ArgumentTypeError("subsets must be 'all' or comma-separated values from 0 to 9")
    return subsets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="data/luna16")
    parser.add_argument("--subsets", type=_parse_subsets, default=list(range(10)))
    parser.add_argument("--no-extract", action="store_true", help="Only download and verify ZIP files.")
    parser.add_argument("--list", action="store_true", help="Show the official file manifest without downloading.")
    args = parser.parse_args()

    total_size = sum(ZENODO_RECORDS[index][2] for index in args.subsets) + ANNOTATIONS[2]
    if args.list:
        for index in args.subsets:
            print(f"subset{index}.zip\t{ZENODO_RECORDS[index][2] / 1024**3:.3f} GiB")
        print(f"annotations.csv\t{ANNOTATIONS[2] / 1024**2:.3f} MiB")
        print(f"total\t{total_size / 1024**3:.3f} GiB")
        return 0

    output_root = Path(args.output_root)
    downloads = output_root / "downloads"
    record_id, checksum, size = ANNOTATIONS
    _download(_zenodo_url(record_id, "annotations.csv"), output_root / "annotations.csv", size, checksum)

    for index in args.subsets:
        record_id, checksum, size = ZENODO_RECORDS[index]
        filename = f"subset{index}.zip"
        archive = _download(_zenodo_url(record_id, filename), downloads / filename, size, checksum)
        if not args.no_extract:
            _extract_subset(archive, output_root, index)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Download interrupted; partial file was kept for resume.", file=sys.stderr)
        raise SystemExit(130)
