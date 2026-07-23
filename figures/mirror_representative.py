"""Mirror-class representatives for moral systems.

A moral system ``(p, d)`` and its Good/Bad relabelling -- its *mirror*
``(p_hat, d_hat)`` -- are behaviourally equivalent: they carry identical
cooperation and stability indices, and their resident reputation equilibria are
``g*`` and ``1 - g*``. Sweeping the whole space therefore double-counts every
non-self-mirror system. This module selects exactly one *representative* per
mirror class, following the manuscript's Definition (Majority-good
representatives):

    The representative of a mirror class is
      * its unique member, if the class is a self-mirror;
      * otherwise, the member with g* > 1/2, if there is one;
      * otherwise (both members sit at g* = 1/2), the member with the larger
        strategy ID, or -- if the two members share the same strategy -- the
        member whose assessment rule has the larger ID.

Bit encoding (matching ``src/MoralConsistency.jl``):
  * p bit for (own rep i, opp rep j)            -> position  j + 2i
  * d bit for (own rep i, opp rep j, own act k, opp act l) -> position l + 2k + 4j + 8i
IDs are the little-endian integers p in 0..15 and d in 0..65535.

Mirror map (Good<->Bad relabelling):
  * p_hat(i, j)       = p(1-i, 1-j)
  * d_hat(i, j, k, l) = NOT d(1-i, 1-j, k, l)

The two involutions are precomputed into lookup tables so the whole selection
is a handful of vectorised NumPy operations over the ~1M-row frame.
"""

import numpy as np

# Genuine g* = 1/2 ties are exact in theory; the CSV stores a solved float, so
# treat anything within this tolerance of 1/2 as a tie and fall through to the
# deterministic ID tie-break. Real majority-good/-bad systems sit far further
# from 1/2 than this at the error rates studied, so no non-tie is misclassified.
HALF_TOL = 1e-9


def _pbit(p, i, j):
    return (p >> (j + 2 * i)) & 1


def _dbit(d, i, j, k, l):
    return (d >> (l + 2 * k + 4 * j + 8 * i)) & 1


def _build_mirror_p():
    """Lookup table: MIRROR_P[p] = mirror of strategy p, over p in 0..15."""
    table = np.empty(16, dtype=np.int64)
    for p in range(16):
        q = 0
        for i in (0, 1):
            for j in (0, 1):
                q |= _pbit(p, 1 - i, 1 - j) << (j + 2 * i)
        table[p] = q
    return table


def _build_mirror_d():
    """Lookup table: MIRROR_D[d] = mirror of assessment rule d, over 0..65535.

    Built vectorised: for each of the 16 argument profiles, copy the flipped
    source bit into its destination position across the whole 0..65535 range.
    """
    d = np.arange(1 << 16, dtype=np.int64)
    q = np.zeros_like(d)
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                for l in (0, 1):
                    src_pos = l + 2 * k + 4 * (1 - j) + 8 * (1 - i)
                    dst_pos = l + 2 * k + 4 * j + 8 * i
                    src_bit = (d >> src_pos) & 1
                    q |= (1 - src_bit) << dst_pos
    return q


MIRROR_P = _build_mirror_p()
MIRROR_D = _build_mirror_d()


def mirror_ids(p, d):
    """Return (p_hat, d_hat), the mirror of each (p, d) pair (array-aware)."""
    p = np.asarray(p, dtype=np.int64)
    d = np.asarray(d, dtype=np.int64)
    return MIRROR_P[p], MIRROR_D[d]


def representative_mask(p, d, g_star, tol=HALF_TOL):
    """Boolean mask: True where (p, d) is the representative of its mirror class.

    Parameters
    ----------
    p, d : array-like of int
        Strategy IDs (0..15) and assessment-rule IDs (0..65535).
    g_star : array-like of float
        Resident reputation equilibrium of each (p, d) -- the CSV column
        ``resident_reputation_eq``. Only the system's own g* is needed: the
        mirror's is ``1 - g*``, so ``g* > 1/2`` already picks the majority-good
        member of every non-self-mirror pair.
    tol : float
        Half-window around 1/2 treated as a g* = 1/2 tie.
    """
    p = np.asarray(p, dtype=np.int64)
    d = np.asarray(d, dtype=np.int64)
    g = np.asarray(g_star, dtype=float)

    p_hat = MIRROR_P[p]
    d_hat = MIRROR_D[d]

    self_mirror = (p_hat == p) & (d_hat == d)
    majority_good = g > 0.5 + tol
    tie = np.abs(g - 0.5) <= tol
    # ID tie-break: larger strategy ID, or larger assessment-rule ID when the
    # two members share the same (self-mirror) strategy.
    tie_win = (p > p_hat) | ((p == p_hat) & (d > d_hat))

    return self_mirror | majority_good | (tie & tie_win)


def filter_representatives(df, p_col='p', d_col='d',
                           g_col='resident_reputation_eq', tol=HALF_TOL):
    """Return the sub-frame of ``df`` holding one representative per mirror class."""
    for col in (p_col, d_col, g_col):
        if col not in df.columns:
            raise ValueError(
                f"filter_representatives: required column {col!r} not found; "
                f"present columns: {list(df.columns)}"
            )
    mask = representative_mask(df[p_col].values, df[d_col].values,
                               df[g_col].values, tol=tol)
    return df.loc[mask].copy()
