#!/usr/bin/env python3
"""From-scratch resumable DRL teacher training."""
from pathlib import Path
import runpy
if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parent/'src/train_corrected_teacher.py'), run_name='__main__')
