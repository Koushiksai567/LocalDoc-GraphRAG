#!/usr/bin/env python3
"""Download and snapshot current official U.S. immigration sources for GraphRAG.

Run from the GraphRAG project root:
    ./.venv/bin/python scripts/download_immigration_sources.py --force

The script uses only official government URLs listed in
config/immigration_sources.json. Direct PDFs are downloaded with curl and live
webpages are printed to searchable PDFs with Chrome/Chromium.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_pdf(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 100 and path.read_bytes()[:5] == b"%PDF-"
    except OSError:
        return False


def find_chrome() -> str | None:
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def download_pdf(url: str, output: Path) -> None:
    curl = shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is required but was not found")

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output.with_suffix(output.suffix + ".part")
    temp_path.unlink(missing_ok=True)
    command = [
        curl,
        "--location",
        "--fail",
        "--silent",
        "--show-error",
        "--retry", "3",
        "--retry-delay", "2",
        "--connect-timeout", "30",
        "--max-time", "180",
        "--user-agent", "Mozilla/5.0 ImmigrationGraphRAGDataset/1.0",
        "--output", str(temp_path),
        url,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(result.stderr.strip() or "curl download failed")
    if not is_pdf(temp_path):
        temp_path.unlink(missing_ok=True)
        raise RuntimeError("The URL did not return a valid PDF")
    temp_path.replace(output)


def snapshot_webpage(url: str, output: Path, chrome: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    profile = tempfile.mkdtemp(prefix="immigration-kb-chrome-")
    try:
        command = [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-extensions",
            f"--user-data-dir={profile}",
            f"--print-to-pdf={output}",
            "--print-to-pdf-no-header",
            "--virtual-time-budget=15000",
            url,
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Chrome export failed")
        if not is_pdf(output):
            raise RuntimeError("Chrome did not create a valid PDF")
    finally:
        shutil.rmtree(profile, ignore_errors=True)


def write_manifests(output_root: Path, records: list[dict[str, Any]]) -> None:
    json_path = output_root / "source_manifest.json"
    csv_path = output_root / "source_manifest.csv"
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    columns = [
        "id", "title", "agency", "topic", "kind", "source_url", "local_file",
        "retrieved_at", "status", "sha256", "size_bytes", "notes"
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({column: record.get(column, "") for column in columns})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", help="GraphRAG project root; default is current directory")
    parser.add_argument("--force", action="store_true", help="Replace all existing snapshots")
    parser.add_argument("--pdf-only", action="store_true", help="Download direct PDFs but skip live webpage snapshots")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    registry_path = project_root / "config" / "immigration_sources.json"
    if not registry_path.exists():
        print(f"ERROR: Missing source registry: {registry_path}", file=sys.stderr)
        print("Run this command from the project root after extracting the kit.", file=sys.stderr)
        return 2

    output_root = project_root / "data" / "immigration"
    output_root.mkdir(parents=True, exist_ok=True)
    sources = json.loads(registry_path.read_text(encoding="utf-8"))
    chrome = None if args.pdf_only else find_chrome()
    if not args.pdf_only and chrome is None:
        print("WARNING: Chrome/Chromium was not found. Webpage snapshots will be skipped.")

    now = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []
    failures: list[str] = []

    for source in sources:
        configured_output = Path(source["output"])
        # The registry is shared with a standalone kit; always place documents
        # below the active project's data/immigration directory.
        try:
            relative_after_immigration = configured_output.parts[configured_output.parts.index("immigration") + 1 :]
        except ValueError:
            relative_after_immigration = (configured_output.name,)
        local_path = output_root.joinpath(*relative_after_immigration)

        record: dict[str, Any] = {
            "id": source["id"],
            "title": source["title"],
            "agency": source["agency"],
            "topic": source["topic"],
            "kind": source["kind"],
            "source_url": source["url"],
            "local_file": str(local_path.relative_to(project_root)),
            "retrieved_at": now,
            "status": "pending",
            "sha256": "",
            "size_bytes": "",
            "notes": "",
        }

        try:
            if local_path.exists() and is_pdf(local_path) and not args.force:
                record["status"] = "existing"
            elif source["kind"] == "pdf":
                print(f"Downloading PDF: {source['title']}")
                download_pdf(source["url"], local_path)
                record["status"] = "downloaded"
            elif source["kind"] == "webpage":
                if args.pdf_only:
                    record["status"] = "skipped"
                    record["notes"] = "Skipped because --pdf-only was selected."
                elif chrome is None:
                    record["status"] = "skipped"
                    record["notes"] = "Chrome/Chromium not found; save the official URL as PDF manually."
                else:
                    print(f"Creating live webpage snapshot: {source['title']}")
                    snapshot_webpage(source["url"], local_path, chrome)
                    record["status"] = "downloaded"
            else:
                raise RuntimeError(f"Unsupported kind: {source['kind']}")

            if is_pdf(local_path):
                record["sha256"] = sha256_file(local_path)
                record["size_bytes"] = local_path.stat().st_size
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
            local_path.unlink(missing_ok=True)
            record["status"] = "failed"
            record["notes"] = str(exc)
            failures.append(f"{source['id']} | {source['url']} | {exc}")
            print(f"FAILED: {source['title']} - {exc}", file=sys.stderr)

        records.append(record)

    write_manifests(output_root, records)
    failed_path = output_root / "failed_sources.txt"
    if failures:
        failed_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
    else:
        failed_path.unlink(missing_ok=True)

    ready = sum(record["status"] in {"downloaded", "existing"} for record in records)
    failed = sum(record["status"] == "failed" for record in records)
    skipped = sum(record["status"] == "skipped" for record in records)
    print("\nFinished")
    print(f"Ready documents: {ready}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Documents: {output_root}")
    print(f"Manifest: {output_root / 'source_manifest.csv'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
