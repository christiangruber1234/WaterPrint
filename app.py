#!/usr/bin/env python3
"""
WaterPrint Web Application - HuggingFace Spaces Entry Point

This file is the entry point for HuggingFace Spaces deployment.
It imports and launches the webapp from src/webapp.py.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from webapp import create_interface

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(ssr_mode=False)
