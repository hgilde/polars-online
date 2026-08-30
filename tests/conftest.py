import sys
from pathlib import Path

# Make tests/data.py and tests/reference.py importable as plain modules.
sys.path.insert(0, str(Path(__file__).resolve().parent))
