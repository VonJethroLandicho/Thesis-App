"""
build_sample_bank.py
====================
Root convenience wrapper for data_pipeline/08_build_sample_bank.py.
Delegates to the official data pipeline Step 08.
"""

import sys
from pathlib import Path

# Add data_pipeline to sys.path
pipeline_dir = Path(__file__).resolve().parent / "data_pipeline"
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

import importlib.util

spec = importlib.util.spec_from_file_location("build_sample_bank_step8", pipeline_dir / "08_build_sample_bank.py")
step8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step8)

if __name__ == "__main__":
    step8.main()
