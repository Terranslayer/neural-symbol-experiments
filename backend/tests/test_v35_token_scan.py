# -*- coding: utf-8 -*-
"""Zero-Prior guard (spec 2026-06-05 §9): the NEW V35 PSS leak-kill identifiers must
not match any forbidden human-number-system token. (The whole-repo scan is NOT green
today — design notes contain the banned words as meta-discussion — so the enforceable
guarantee is identifier-level, checked here.)
"""
import re
from backend.tests.validate_constraints import FORBIDDEN_PRIOR_PATTERNS

NEW_V35_IDENTIFIERS = [
    # scene.py
    "successor_balanced_preset", "beta_range_preset", "scale_weight",
    "successor_far_bands", "THETA",
    # mamba_agent.py
    "v35_pss_readback_beta", "_scratch_beta_for_aux", "_scratch_beta_raw",
    "_v35_residuals_beta", "_v35_lesion",
    # train_phase1.py
    "balanced_n_classes", "k_to_class", "inverse_freq_class_weights",
]


def test_new_identifiers_are_zero_prior_clean():
    for ident in NEW_V35_IDENTIFIERS:
        for pat in FORBIDDEN_PRIOR_PATTERNS:
            assert not re.search(pat, ident, flags=re.IGNORECASE), (
                f"new identifier {ident!r} matches forbidden Zero-Prior pattern {pat!r} "
                f"— rename it (e.g. avoid 'base'+digit / 'radix' / 'binary' / 'decimal')"
            )
