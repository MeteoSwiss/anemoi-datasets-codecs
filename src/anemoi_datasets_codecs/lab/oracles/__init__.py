"""lab oracles: offline estimates, in bits per cell, of what an idea would buy before anyone builds it.

Each oracle runs on one date of a store with a lattice codec configuration (``anemoi-datasets-codecs lab oracles
ORACLE CONFIG STORE [--date I] [--fields A,B] [--json]``) and on arrays from Python. Fields are indices
within a date; names only label the output. Common ground (:mod:`.core`): every lossy field is quantised by
``encoder.quantise`` with the configuration's ``k`` and clip and the **plain rule** (``recon_bits`` 0, not the
default residual-domain rule) and predicted **open loop** from the quantised values along the configuration's
structure (with the plain rule the decoder reconstructs exactly these values, so this is the codec's prediction);
costs are :func:`lab.entropy.residual_cost_bits`: symbol entropy conditional on a context plus 16 bits per escape.

* ``baseline``: the codec's spatial prediction (default weights ``w0``, the encoder's robust sum-to-one fit, and
  its per-step choice between them by order-0 cost) with the codec's context (spread of the first <= 4 sources,
  8 per-step quantile buckets). Calibration: the same fields encoded by ``LatticeCodec`` with ``recon_bits`` 0 and
  the physics layer off (the coding modelled here), bytes from ``container.inspect_chunk``. The oracle is expected a
  few percent below the coded size: table bytes, the rANS precision, range flushes, the per-field header and the
  escapes' actual byte cost are not modelled, and the codec may pick a cheaper table layout than per-step contexts.
  ``--calibrate full`` also reports the configuration's own codec on the whole date (residual-domain rule, cross
  terms), which the baseline does not model.
* ``linear``: the unconstrained linear predictor over the step's sources with an intercept (robust fit), optionally
  with the nearest source's nearest source, against the sum-to-one weights: the headroom of coefficient-carrying
  predictors with the same stencil.
* ``cross``: every ordered pair of lossy fields: the gain of a one-coefficient regression of the target's residual
  on the source's residual per step, and of conditioning on the source's residual bucket, with the codec's plan
  terms marked (:mod:`.cross`).
* ``temporal``: the previous date of the store as a predictor: spatial prediction of the difference, a robust
  per-step mix of the spatial prediction and the previous value; only meaningful on consecutive dates.
* ``contexts``: conditional entropy of the codec's symbols under richer contexts (32 spread buckets, the nearest
  source's residual magnitude, both, the sum over the sources) and the symbol stream through zstd -19 / xz -9e as a
  cheap bound of adaptive modelling; table bytes reported, not charged.
* ``levels``: per step (refinement level) the targets, the variance of data and residuals, and the bits: where the
  cost sits in the multiresolution hierarchy.

Every estimate ignores the bytes a new model would carry (coefficients, tables) unless stated, and the closed-loop
effects of residual-domain reconstruction. Nothing here is on the encode or decode path.
"""

from __future__ import annotations

from .cli import add_parsers, run_baseline, run_contexts, run_levels, run_linear, run_temporal
from .core import FieldSpec, Inputs, Spatial, cost_bits, quantised, requantised, select_fields, spatial
from .cross import cross

__all__ = [
    "FieldSpec",
    "Inputs",
    "Spatial",
    "add_parsers",
    "cost_bits",
    "cross",
    "quantised",
    "requantised",
    "run_baseline",
    "run_contexts",
    "run_levels",
    "run_linear",
    "run_temporal",
    "select_fields",
    "spatial",
]
