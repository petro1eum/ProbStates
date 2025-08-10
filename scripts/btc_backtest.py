#!/usr/bin/env python3
import numpy as np
import pandas as pd
import yfinance as yf
from probstates.markets import sma, momentum, rsi, indicator_to_prob, FeatureSpec, aggregate_specs
from probstates import set_phase_or_mode


def load_btc(start='2018-01-01', end='2025-01-01'):
    df = yf.download('BTC-USD', start=start, end=end, progress=False)
    df = df[['Close']].dropna().copy()
    df['mom10'] = momentum(df['Close'].values, 10)
    df['rsi14'] = rsi(df['Close'].values, 14)
    df['sma20'] = sma(df['Close'].values, 20)
    df['sma50'] = sma(df['Close'].values, 50)
    df.dropna(inplace=True)
    return df


def build_specs(row):
    specs = []
    p_mom = float(indicator_to_prob(np.array([row['mom10']]))[-1])
    phi_mom = 0.0 if row['mom10'] >= 0 else np.pi
    specs.append(FeatureSpec('momentum_10', p_mom, phi_mom, 2))
    p_rsi = float(indicator_to_prob(np.array([row['rsi14']]), center=50.0, k=1.2)[-1])
    phi_rsi = 0.0 if row['rsi14'] >= 50 else np.pi
    specs.append(FeatureSpec('rsi_14', p_rsi, phi_rsi, 1))
    cross = float(row['sma20'] - row['sma50'])
    p_cross = float(indicator_to_prob(np.array([cross]))[-1])
    phi_cross = 0.0 if cross >= 0 else np.pi
    specs.append(FeatureSpec('sma20_vs_50', p_cross, phi_cross, 2))
    return specs


def backtest(df: pd.DataFrame, buy_thr=0.55, sell_thr=0.45, fee=0.001):
    set_phase_or_mode('weight')
    agg_p = []
    for _, r in df.iterrows():
        agg_p.append(aggregate_specs(build_specs(r), mode='weight').probability)
    df = df.copy()
    df['agg_p'] = agg_p
    pos = 0.0
    positions = []
    for p in df['agg_p'].values:
        if p >= buy_thr:
            pos = 1.0
        elif p <= sell_thr:
            pos = 0.0
        positions.append(pos)
    df['pos'] = positions
    ret = df['Close'].pct_change().fillna(0.0)
    trade = np.abs(np.diff([0.0, *df['pos'].values]))
    strat = df['pos'].shift(1).fillna(0.0) * ret - fee * trade
    return df, strat


def metrics(series: pd.Series):
    x = series.dropna().values
    ann = np.prod(1.0 + x) ** (252/len(x)) - 1.0
    vol = np.std(x) * np.sqrt(252)
    sharpe = ann / (vol + 1e-12)
    curve = (1.0 + x).cumprod()
    peak = np.maximum.accumulate(curve)
    mdd = float(np.max((peak - curve)/peak))
    return dict(CAGR=ann, Sharpe=sharpe, MDD=mdd)


def main():
    df = load_btc()
    df, strat = backtest(df)
    bench = df['Close'].pct_change().fillna(0.0)
    print('Strategy:', {k: round(v,3) for k,v in metrics(strat).items()})
    print('Buy&Hold:', {k: round(v,3) for k,v in metrics(bench).items()})


if __name__ == '__main__':
    main()


