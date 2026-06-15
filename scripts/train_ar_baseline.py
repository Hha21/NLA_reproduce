"""
Phase 2 — train the Reconstructor (AR) on ground-truth text.

This establishes the oracle ceiling: the best FVE achievable when the
description is the original snippet itself (i.e. perfect information).
All later NLA results must stay below this ceiling.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
