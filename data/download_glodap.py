#!/usr/bin/env python3
"""
Download GLODAP v2.2023 dataset.

Usage:
    python data/download_glodap.py
"""

import urllib.request
import sys
from pathlib import Path


DATA_DIR = Path(__file__).parent
GLODAP_URL = "https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0283442/GLODAPv2.2023_Merged_Master_File.csv"
OUTPUT_FILE = DATA_DIR / "GLODAPv2.2023_Merged_Master_File.csv"


def download_progress(block_num, block_size, total_size):
    """Show download progress."""
    downloaded = block_num * block_size
    percent = min(100, downloaded * 100 / total_size)
    mb_downloaded = downloaded / (1024 * 1024)
    mb_total = total_size / (1024 * 1024)
    sys.stdout.write(f"\r  Downloading: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
    sys.stdout.flush()


def download_glodap():
    """Download GLODAP v2.2023 Merged Master File."""
    if OUTPUT_FILE.exists():
        size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
        print(f"GLODAP data already exists: {OUTPUT_FILE}")
        print(f"  Size: {size_mb:.1f} MB")
        return OUTPUT_FILE

    print("Downloading GLODAP v2.2023 Merged Master File...")
    print(f"  Source: {GLODAP_URL}")
    print(f"  Target: {OUTPUT_FILE}")

    try:
        urllib.request.urlretrieve(GLODAP_URL, OUTPUT_FILE, download_progress)
        print("\n  Download complete!")
        size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
        print(f"  Size: {size_mb:.1f} MB")
        return OUTPUT_FILE
    except Exception as e:
        print(f"\n  Download failed: {e}")
        print("\n  Please download manually from: https://www.glodap.info/")
        sys.exit(1)


if __name__ == "__main__":
    download_glodap()
