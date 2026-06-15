"""
Activation Verbalizer (AV) — the encoder half of the NLA.

Responsibilities:
  - AV class: injects an activation vector as a soft token, generates a text description
  - Sampling: draw G descriptions per activation (needed for GRPO group advantage)
  - GRPO training step: policy-gradient loss + KL penalty against a frozen reference copy
"""
