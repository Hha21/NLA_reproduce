"""
Phase 5 — full NLA training with GRPO.

Jointly trains:
  - AV via policy gradient (reward = cosine similarity of reconstructed activation)
  - AR via supervised regression on AV's sampled descriptions
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
