"""
Activation extraction and dataset utilities.

Responsibilities:
  - Register the forward hook on the frozen target T
  - get_activation(text) → residual-stream vector at PROBE_LAYER
  - Build and cache a dataset of (text, activation) pairs to disk
"""
