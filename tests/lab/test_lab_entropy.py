import numpy as np

from anemoi_datasets_codecs.lab import entropy


def test_symbols_match_codec_mapping():
    r = np.array([0, -1, 1, -2, 2, 127, -128, 200])
    assert entropy.zigzag(r).tolist() == [0, 1, 2, 3, 4, 254, 255, 400]
    assert entropy.symbols(r).tolist() == [0, 1, 2, 3, 4, 254, 255, 255]


def test_entropy_values():
    assert entropy.entropy_bits(np.zeros(100, np.int64)) == 0.0
    assert abs(entropy.entropy_bits(np.arange(8)) - 24.0) < 1e-9  # 8 symbols, 3 bits each
    sym = np.array([0, 0, 1, 1])
    assert entropy.conditional_entropy_bits(sym, np.array([0, 0, 1, 1])) == 0.0
    assert abs(entropy.conditional_entropy_bits(sym, np.zeros(4)) - 4.0) < 1e-9
    assert entropy.escape_bits(np.array([127, 128, -128])) == 32.0


def test_laplace_fit():
    rng = np.random.default_rng(0)
    r = np.rint(rng.laplace(0, 10, 100000))
    loc, scale = entropy.laplace_fit(r)
    assert abs(loc) < 0.5 and abs(scale - 10) < 0.3
