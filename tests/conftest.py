import sys
from pathlib import Path

# make repo root importable regardless of where pytest is invoked from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
