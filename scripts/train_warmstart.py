"""
Phase 4 — supervised warm-start for the Verbalizer (AV).

SFT on (activation → original snippet text) pairs before RL begins.
Without this, AV ignores the activation and RL reward is pure noise.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
