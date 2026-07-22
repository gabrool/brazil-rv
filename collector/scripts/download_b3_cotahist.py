from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

SCRIPT_VERSION = "1"
DEFAULT_BASE_URL = "https://bvmf.bmfbovespa.com.br/InstDados/SerHist"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36"
)


@dataclass
class DownloadResult:
    year: int
    filename: str
    url: str
    status: str
    bytes: int = 0
    sha256: str = ""
    zip_member: str = ""
    error: str = ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_zip(path: Path, expected_year: int) -> str:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("Downloaded file is missing or empty")
    if not zipfile.is_zipfile(path):
        preview = path.read_bytes()[:200]
        raise ValueError(
            "Response is not a ZIP file; B3 may have returned an HTML/CAPTCHA page. "
            f"First bytes: {preview!r}"
        )
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.upper().endswith(".TXT")]
        if not members:
            raise ValueError("ZIP contains no TXT member")
        preferred = [name for name in members if str(expected_year) in Path(name).name and "COTAHIST" in name.upper()]
        member = preferred[0] if preferred else members[0]
        info = archive.getinfo(member)
        if info.file_size < 245 * 3:
            raise ValueError(f"TXT member is unexpectedly small: {info.file_size} bytes")
        return member


def download_one(*, year: int, out_dir: Path, base_url: str, overwrite: bool, timeout: float, retries: int, insecure: bool) -> DownloadResult:
    filename = f"COTAHIST_A{year}.ZIP"
    destination = out_dir / filename
    url = f"{base_url.rstrip('/')}/{filename}"
    if destination.exists() and not overwrite:
        try:
            member = validate_zip(destination, year)
            return DownloadResult(year, filename, url, "existing_valid", destination.stat().st_size, sha256_file(destination), member)
        except Exception as exc:  # noqa: BLE001
            return DownloadResult(year, filename, url, "existing_invalid", destination.stat().st_size, error=str(exc))

    context = ssl._create_unverified_context() if insecure else ssl.create_default_context()
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    last_error = ""
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(url, headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/zip,application/octet-stream,*/*",
            "Referer": "https://www.b3.com.br/",
        })
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
                with temporary.open("wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            member = validate_zip(temporary, year)
            temporary.replace(destination)
            return DownloadResult(year, filename, url, "downloaded", destination.stat().st_size, sha256_file(destination), member)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, OSError) as exc:
            last_error = f"attempt {attempt}/{retries}: {type(exc).__name__}: {exc}"
            temporary.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2 ** attempt, 10))
    return DownloadResult(year, filename, url, "failed", error=last_error)


def parse_years(values: Iterable[str]) -> list[int]:
    years: list[int] = []
    for value in values:
        if "-" in value:
            start_text, end_text = value.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise argparse.ArgumentTypeError(f"Invalid year range: {value}")
            years.extend(range(start, end + 1))
        else:
            years.append(int(value))
    return sorted(set(years))


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and validate official B3 annual COTAHIST ZIP files.")
    parser.add_argument("--years", nargs="+", default=["2021-2026"], help="Years or inclusive ranges")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--insecure", action="store_true", help="Disable TLS verification only if the B3 TLS path is broken")
    args = parser.parse_args()

    years = parse_years(args.years)
    args.out.mkdir(parents=True, exist_ok=True)
    results: list[DownloadResult] = []
    for year in years:
        print(f"Downloading/validating COTAHIST {year} ...", flush=True)
        result = download_one(year=year, out_dir=args.out, base_url=args.base_url, overwrite=args.overwrite, timeout=args.timeout, retries=max(args.retries, 1), insecure=args.insecure)
        results.append(result)
        if result.status in {"downloaded", "existing_valid"}:
            print(f"  {result.status}: {result.filename} ({result.bytes / 1024 / 1024:.1f} MiB; member={result.zip_member})")
        else:
            print(f"  FAILED: {result.error}", file=sys.stderr)

    payload = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "results": [asdict(result) for result in results],
    }
    (args.out / "download_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (args.out / "download_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader(); writer.writerows(asdict(result) for result in results)

    failed = [result for result in results if result.status not in {"downloaded", "existing_valid"}]
    if failed:
        print("\nOne or more direct downloads failed.", file=sys.stderr)
        print(f"Manually download the annual ZIP(s) from the official B3 Historical Quotes page and place them in: {args.out}", file=sys.stderr)
        print("Expected names: " + ", ".join(result.filename for result in failed), file=sys.stderr)
        return 2
    print(f"\nAll requested files are valid: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
