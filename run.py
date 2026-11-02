#!/usr/bin/env python3
"""Run PPL-TSC from a JSON configuration."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ppl_tsc.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
