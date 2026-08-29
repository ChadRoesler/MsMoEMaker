# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors
#
# Vendored from p-e-w/heretic and trimmed to the ablation core. Dropped the
# accelerator/driver/env-reporting helpers (they pulled py-cpuinfo, psutil,
# importlib.metadata and accelerate 1.x-only accelerator flags).

import gc

import torch


def empty_cache():
    """Clears the backend cache and collects garbage."""

    # Collecting garbage is not an idempotent operation, and to avoid OOM errors,
    # gc.collect() has to be called both before and after emptying the backend cache.
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()
