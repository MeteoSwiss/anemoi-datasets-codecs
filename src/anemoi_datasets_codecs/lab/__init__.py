"""The research toolkit of the codecs (``anemoi-datasets-codecs lab ...``).

Tools that say where the bits go (``account``, ``residuals``), what an idea would buy before it is built
(``oracles``), how the codec stands against reference compressors at the same bound (``refs``), and that keep
experiments comparable (``ledger``, the ``minigrid`` cut-out of a grid with real data) and the decoder honest
(``fuzz``). Nothing here is on the encode or decode path of a store: the toolkit imports the engine, the engine never
imports the toolkit, and no tool changes a frozen format. Shared pieces: ``data`` (stores, configurations, dates,
field labels, the environment of a run) and ``entropy`` (entropy estimates of residual symbols).
"""
