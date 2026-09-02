"""
mt_utils.py -- Mini-Transformer Shared Utilities
MATH/DATA 412, University of Arizona

This module grows across all 15 coding assignments.
Students add to it each week; by HW 12 it is a complete
mini-transformer forward pass.

Copyright 2026 Jason Aubrey / University of Arizona. All rights reserved.
"""

import numpy as np


# ── HW 1: Scaffolding ──────────────────────────────────────────────────────

SEED = 412  # reproducibility throughout the course


def set_seed(seed=SEED):
    """Set NumPy random seed for reproducibility."""
    np.random.seed(seed)


def make_embedding_matrix(vocab_size, d_model, seed=SEED):
    """
    Create a random token embedding matrix E of shape (vocab_size, d_model).
    Row i is the embedding vector for token i.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((vocab_size, d_model)) * 0.02


def embed_tokens(token_ids, E):
    """
    Look up embeddings for a list of token IDs.
    Returns X of shape (n_tokens, d_model).
    """
    return E[token_ids]


# ── HW 2 onward: functions added by students ───────────────────────────────
# Each HW adds the functions developed in that week's milestone section.
# By HW 12, this file contains the full forward pass.
