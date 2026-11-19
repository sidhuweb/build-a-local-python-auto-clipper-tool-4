#!/usr/bin/env python3
"""
AutoClipper standalone runner script.
Run:
    python autoclipper.py "https://youtube.com/..."
    python autoclipper.py --demo
"""
import sys
from autoclipper.cli import main

if __name__ == "__main__":
    main()
