import numpy as np

from probstates.markets import (
    sma, momentum, rsi,
    indicator_to_prob, sentiment_to_phase,
    FeatureSpec, make_phase_states, aggregate_specs,
    btc_signal_from_arrays,
)


def test_indicators_shapes():
    x = np.arange(1, 101, dtype=float)
    assert np.isfinite(sma(x, 20)[-1])
    assert np.isfinite(momentum(x, 10)[-1])
    assert np.isfinite(rsi(x, 14)[-1])


def test_mapping_and_aggregation():
    probs = indicator_to_prob(np.array([ -1.0, 0.0, 1.0 ]))
    assert (probs >= 0).all() and (probs <= 1).all()
    phi_pos = sentiment_to_phase(+0.8)
    phi_neg = sentiment_to_phase(-0.8)
    assert 0.0 <= phi_pos <= np.pi and 0.0 <= phi_neg <= np.pi

    specs = [
        FeatureSpec('a', 0.7, 0.0, 2.0),
        FeatureSpec('b', 0.6, np.pi, 1.0),
    ]
    states = make_phase_states(specs)
    agg = aggregate_specs(specs, mode='weight')
    assert 0.0 <= agg.probability <= 1.0


def test_btc_signal_pipeline():
    rng = np.random.default_rng(0)
    # synthetic upward trend
    closes = np.cumsum(rng.normal(0.1, 1.0, size=120)) + 30000.0
    news = [0.0, +0.5, +0.8]
    agg, decision = btc_signal_from_arrays(closes, news_sentiments=news, mode='weight')
    assert 0.0 <= agg.probability <= 1.0
    assert decision in ('buy','sell','hold')


