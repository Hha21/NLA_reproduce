"""
Activation Reconstructor (AR) — the decoder half of the NLA.

Responsibilities:
  - AR class: takes a text description, returns a predicted activation vector â
  - Loss: MSE or cosine between â and the target activation a
  - Supervised training step (plain gradient descent, no RL needed)
"""
