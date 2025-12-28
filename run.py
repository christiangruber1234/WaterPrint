#!/usr/bin/env python3
"""
WaterPrint: Complete Analysis Pipeline

Downloads data, runs all analyses, and generates figures.

Usage:
    python run.py
"""

import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
SRC_DIR = PROJECT_DIR / "src"


def run_script(script_path, description):
    """Run a Python script and handle errors."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")

    result = subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_DIR)

    if result.returncode != 0:
        print(f"\nError running {script_path.name}")
        sys.exit(1)


def main():
    print("="*60)
    print("  WATERPRINT: COMPLETE ANALYSIS PIPELINE")
    print("="*60)

    # Step 1: Download data
    run_script(DATA_DIR / "download_glodap.py", "Step 1/3: Downloading GLODAP data")

    # Step 2: Run isotope classification analysis
    run_script(SRC_DIR / "isotope_classification.py", "Step 2/3: Isotope classification analysis")

    # Step 3: Generate all figures
    run_script(SRC_DIR / "generate_figures.py", "Step 3/3: Generating figures")

    print("\n" + "="*60)
    print("  COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {DATA_DIR}")
    print(f"Figures saved to: {PROJECT_DIR / 'manuscript' / 'revision_5' / 'figures'}")


if __name__ == "__main__":
    main()
