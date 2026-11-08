# SPDX-License-Identifier: BSD-3-Clause
# Adapted from scikit-image (https://github.com/scikit-image/scikit-image),
# src/skimage/color/delta_e.py (deltaE_ciede2000) and
# src/skimage/color/colorconv.py (_cart2polar_2pi), at commit
# 5d91fed804ec8d9133118f02b3a785790df75934.
# Copyright 2009-2022 the scikit-image team, BSD-3-Clause; the licence is
# LICENSE.scikit-image in the skill's folder. Copied unchanged from the hub's
# tools/colour/_ciede2000.py (which made the changes below), so the skill does
# not import a tool bundle's internals.
#
# Changes from upstream: float64 only (upstream's _float_inputs dtype
# promotion dropped), channels on the last axis only (channel_axis removed),
# inputs broadcast against each other (so one colour can be compared with a
# whole palette), _cart2polar_2pi inlined, docstring shortened. The arithmetic
# is upstream's, line for line.
"""CIEDE2000 colour difference (CIE 142-2001), numpy only."""

from __future__ import annotations

import numpy as np

__all__ = ["deltaE_ciede2000"]


def _cart2polar_2pi(x, y):
    """Convert cartesian coordinates to polar (r, theta) with theta in [0, 2pi)."""
    r, t = np.hypot(x, y), np.arctan2(y, x)
    t += np.where(t < 0.0, 2 * np.pi, 0)
    return r, t


def deltaE_ciede2000(lab1, lab2, kL=1, kC=1, kH=1):
    """CIEDE2000 difference between Lab colours ``lab1`` and ``lab2`` (last axis L, a, b).

    The two arrays broadcast against each other. ``kL``, ``kC``, ``kH`` are the
    parametric weights for lightness, chroma and hue (1 = reference
    conditions; ``kL=2`` is the textile convention that halves lightness).

    References: G. Sharma, W. Wu, E. N. Dalal, "The CIEDE2000 color-difference
    formula: implementation notes, supplementary test data, and mathematical
    observations", Color Res. Appl. 30(1), 2005.
    """
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    lab1, lab2 = np.broadcast_arrays(lab1, lab2)
    unroll = False
    if lab1.ndim == 1:
        unroll = True
        lab1 = lab1[None, :]
        lab2 = lab2[None, :]
    L1, a1, b1 = np.moveaxis(lab1, source=-1, destination=0)[:3]
    L2, a2, b2 = np.moveaxis(lab2, source=-1, destination=0)[:3]

    # distort `a` based on average chroma
    # then convert to lch coordinates from distorted `a`
    # all subsequence calculations are in the new coordinates
    # (often denoted "prime" in the literature)
    Cbar = 0.5 * (np.hypot(a1, b1) + np.hypot(a2, b2))
    c7 = Cbar**7
    G = 0.5 * (1 - np.sqrt(c7 / (c7 + 25**7)))
    scale = 1 + G
    C1, h1 = _cart2polar_2pi(a1 * scale, b1)
    C2, h2 = _cart2polar_2pi(a2 * scale, b2)

    # lightness term
    Lbar = 0.5 * (L1 + L2)
    tmp = (Lbar - 50) ** 2
    SL = 1 + 0.015 * tmp / np.sqrt(20 + tmp)
    L_term = (L2 - L1) / (kL * SL)

    # chroma term
    Cbar = 0.5 * (C1 + C2)  # new coordinates
    SC = 1 + 0.045 * Cbar
    C_term = (C2 - C1) / (kC * SC)

    # hue term
    h_diff = h2 - h1
    h_sum = h1 + h2
    CC = C1 * C2

    dH = h_diff.copy()
    dH[h_diff > np.pi] -= 2 * np.pi
    dH[h_diff < -np.pi] += 2 * np.pi
    dH[CC == 0.0] = 0.0  # if r == 0, dtheta == 0
    dH_term = 2 * np.sqrt(CC) * np.sin(dH / 2)

    Hbar = h_sum.copy()
    mask = np.logical_and(CC != 0.0, np.abs(h_diff) > np.pi)
    Hbar[mask * (h_sum < 2 * np.pi)] += 2 * np.pi
    Hbar[mask * (h_sum >= 2 * np.pi)] -= 2 * np.pi
    Hbar[CC == 0.0] *= 2
    Hbar *= 0.5

    T = (
        1
        - 0.17 * np.cos(Hbar - np.deg2rad(30))
        + 0.24 * np.cos(2 * Hbar)
        + 0.32 * np.cos(3 * Hbar + np.deg2rad(6))
        - 0.20 * np.cos(4 * Hbar - np.deg2rad(63))
    )
    SH = 1 + 0.015 * Cbar * T

    H_term = dH_term / (kH * SH)

    # hue rotation
    c7 = Cbar**7
    Rc = 2 * np.sqrt(c7 / (c7 + 25**7))
    dtheta = np.deg2rad(30) * np.exp(-(((np.rad2deg(Hbar) - 275) / 25) ** 2))
    R_term = -np.sin(2 * dtheta) * Rc * C_term * H_term

    # put it all together
    dE2 = L_term**2
    dE2 += C_term**2
    dE2 += H_term**2
    dE2 += R_term
    ans = np.sqrt(np.maximum(dE2, 0))
    if unroll:
        ans = ans[0]
    return ans
