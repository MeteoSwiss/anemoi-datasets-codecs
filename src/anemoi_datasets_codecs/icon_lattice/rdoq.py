"""The optional rdoq path of the LATTICE encoder (``options.rdoq``, off by default; the decoder is unchanged).

For the predictable fields (plain coding below ``rdoq_max_bits`` bits per value, when ``k < 2``) the field is also
coded at k = 2 with encoder-chosen integers inside the bound (:func:`rdoq_values`), and the smaller payload wins.
"""

from __future__ import annotations

import numpy as np

from .encoder import (
    EncodedField,
    EncodeError,
    EncodeOptions,
    MaskTerm,
    _sample,
    encode_quantised,
    fit_robust,
    quantise,
    reconstruct,
)
from .structure import Structure


def encode_rdoq(
    x: np.ndarray,
    bound: float,
    clip: tuple | None,
    structure: Structure,
    opts: EncodeOptions,
    cross: list[tuple[int, np.ndarray]] | None,
    want_detail: bool,
    mask: MaskTerm | None,
    plain: EncodedField,
) -> EncodedField:
    """The smaller of ``plain`` (the field coded the plain way) and the rdoq coding."""
    # k < 2: try k = 2 with chosen values on the predictable fields only; k >= 2: chosen values at that k everywhere
    if opts.k < 2 and plain.info["bits"] > opts.rdoq_max_bits:
        return plain
    try:
        q = quantise(x, bound, max(2.0, opts.k), clip)
        n = rdoq_values(x, bound, clip, q[0], q[1], q[2], structure, opts.rdoq_lambda)
        alt = encode_quantised(x, bound, clip, structure, opts, cross, want_detail, (q[0], q[1], n), mask)
    except EncodeError:
        return plain
    alt.info["rdoq"] = True
    plain.info["rdoq_bits"] = alt.info["bits"]
    return alt if len(alt.payload) < len(plain.payload) else plain


def rdoq_values(x, bound, clip, offset, step, n0, structure: Structure, lam: float = 1.0, passes: int = 2):
    """Encoder-chosen integers inside the bound: in decode order, each target takes the admissible n minimising
    ``cost_step(n - pred) + lam ((x - recon(n)) / bound)^2`` (pred from the already chosen values with the step's
    default weights; costs from the step's residual histogram, re-estimated once). The decoder is unchanged."""
    xd = x.astype(np.float64)
    cands = np.stack([n0 - 1, n0, n0 + 1])  # (3, n): admissible where the exact float32 reconstruction is in bound
    dist = np.stack([((reconstruct(offset, step, c, clip).astype(np.float64) - xd) / bound) ** 2 for c in cands])
    ok = dist <= 1.0
    dist = np.where(ok, dist, np.inf)
    n = n0.copy()
    costs: list[np.ndarray | None] = [None] * len(structure.steps)
    weights: dict[int, np.ndarray] = {}
    for _ in range(passes + 1):
        new_costs = []
        for s, st in enumerate(structure.steps):
            t = st.t
            if st.kind == "delta":
                pred = None
                res = n[t] - np.concatenate([[0], n[t][:-1]])
            else:
                V = n[st.src]
                if s not in weights:  # robust LS weights on the plain values (what the final encode will refit)
                    smp = _sample(len(t))
                    D = (V[1:] - V[0])[:, smp].T.astype(np.float64)
                    wf = fit_robust(D, (n[t] - V[0])[smp].astype(np.float64)) if st.k > 1 else np.zeros(0)
                    weights[s] = np.clip(np.round(wf * 256), -(2**14), 2**14).astype(np.int64)
                W = weights[s]
                pred = V[0] + ((W @ (V[1:] - V[0]) + 128) >> 8) if st.k > 1 else V[0]
                if costs[s] is not None:
                    r = cands[:, t] - pred[None, :]
                    j = costs[s][np.clip(r, -1024, 1024) + 1024] + lam * dist[:, t]
                    n[t] = cands[np.argmin(j, axis=0), t]
                res = n[t] - pred
            cnt = np.bincount(np.clip(res, -1024, 1024) + 1024, minlength=2049).astype(np.float64) + 0.02
            new_costs.append(-np.log2(cnt / cnt.sum()))
        costs = new_costs
    return n
