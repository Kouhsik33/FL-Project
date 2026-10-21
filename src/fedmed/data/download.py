"""Reproducible download of the public datasets used in this project.

Each source is a CC BY 4.0 dataset from the UCI Machine Learning Repository that
requires no registration. Archives are verified against the SHA-256 digests that
were recorded when the project was set up (see docs/dataset_assessment.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from fedmed.paths import RAW_DIR


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    archive: str
    sha256: str
    landing_page: str
    doi: str
    license: str
    required_files: tuple[str, ...]


SOURCES: dict[str, Source] = {
    "wdbc": Source(
        name="wdbc",
        url="https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
        archive="wdbc.zip",
        sha256="bc154869ef13f753f9e2b5a17e248cfe1ba4b6721db7c4da9f4880e40b05d3af",
        landing_page="https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        doi="10.24432/C5DW2B",
        license="CC BY 4.0",
        required_files=("wdbc.data", "wdbc.names"),
    ),
    "heart_disease": Source(
        name="heart_disease",
        url="https://archive.ics.uci.edu/static/public/45/heart+disease.zip",
        archive="heart_disease.zip",
        sha256="b17cd273da9ce1caa4710fce80227ea454d4dbf9fcbc8e6a9121672751563adc",
        landing_page="https://archive.ics.uci.edu/dataset/45/heart+disease",
        doi="10.24432/C52P4X",
        license="CC BY 4.0",
        required_files=(
            "processed.cleveland.data",
            "processed.hungarian.data",
            "processed.switzerland.data",
            "processed.va.data",
            "heart-disease.names",
        ),
    ),
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str, raw_dir: Path = RAW_DIR, force: bool = False) -> Path:
    """Download and extract one dataset; return its raw directory."""
    src = SOURCES[name]
    target = raw_dir / name
    target.mkdir(parents=True, exist_ok=True)
    archive = target / src.archive
    if force or not archive.exists():
        print(f"Downloading {src.url}")
        urllib.request.urlretrieve(src.url, archive)  # noqa: S310 (fixed https URL)
    digest = sha256_of(archive)
    if digest != src.sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {archive}: expected {src.sha256}, got {digest}. "
            "The upstream file may have changed; inspect it before using it."
        )
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            # Refuse path traversal in archive members.
            if member.startswith("/") or ".." in Path(member).parts:
                raise RuntimeError(f"Unsafe path in archive: {member}")
        zf.extractall(target)
    missing = [f for f in src.required_files if not (target / f).exists()]
    if missing:
        raise RuntimeError(f"Archive for {name} lacks expected files: {missing}")
    manifest = {
        "name": name,
        "url": src.url,
        "landing_page": src.landing_page,
        "doi": src.doi,
        "license": src.license,
        "archive_sha256": digest,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (target / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"{name}: verified {archive.name} ({digest[:12]}...) -> {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", help=f"subset of {list(SOURCES)} (default: all)")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()
    unknown = set(args.datasets) - set(SOURCES)
    if unknown:
        parser.error(f"unknown datasets {sorted(unknown)}; choose from {list(SOURCES)}")
    for name in args.datasets or list(SOURCES):
        download(name, force=args.force)


if __name__ == "__main__":
    main()
