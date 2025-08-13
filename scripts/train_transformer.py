#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import yfinance as yf

from probstates.markets import (
    momentum, rsi, sma, bollinger, macd, atr, realized_vol,
    indicator_to_prob
)
from probstates.market_cache import fetch_prices_cached
from probstates.transformer import train_transformer, predict_with_model, save_model
from probstates.markets import cross_asset_features
from probstates.markets import FeatureSpec, aggregate_specs
from probstates import set_phase_or_mode


def build_feature_matrix(df: pd.DataFrame, ctx: dict) -> np.ndarray:
    cols = []
    cols.append(momentum(df['Close'].values, 10))
    cols.append(rsi(df['Close'].values, 14))
    cols.append(sma(df['Close'].values, 20))
    cols.append(sma(df['Close'].values, 50))
    pb = bollinger(df['Close'].values, 20)[3]
    cols.append(pb)
    cols.append(macd(df['Close'].values)[2])
    cols.append(atr(df['High'].values, df['Low'].values, df['Close'].values, 14))
    cols.append(realized_vol(df['Close'].values, 20))
    if ctx:
        feats = cross_asset_features(df['Close'].values, ctx, windows=(20,60))
        for k in sorted(feats.keys()):
            cols.append(feats[k])
    X = np.vstack(cols).T
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2018-01-01')
    ap.add_argument('--end', default='2025-01-01')
    ap.add_argument('--ctx', default='DX-Y.NYB,^VIX,SPY,GLD,ETH-USD')
    ap.add_argument('--seq', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--mode', default='weight')
    args = ap.parse_args()

    df = yf.download('BTC-USD', start=args.start, end=args.end, progress=False)[['Close','High','Low']].dropna()
    tickers = [t.strip() for t in (args.ctx or '').split(',') if t.strip()]
    ctx_raw = fetch_prices_cached(tickers, args.start, args.end, column='Close') if tickers else {}
    ctx = {k: (v.values if hasattr(v, 'values') else v) for k, v in ctx_raw.items()}

    X = build_feature_matrix(df, ctx)
    y = df['Close'].pct_change().shift(-1).fillna(0.0).values  # next-period return
    # align NaNs
    mask = ~np.any(np.isnan(X), axis=1)
    X = X[mask]
    y = y[mask]

    model = train_transformer(X, y, seq_len=args.seq, epochs=args.epochs, mode=args.mode)
    save_model(model, 'data/models/transformer.pt')

    # quick sanity prediction
    preds, specs_list = predict_with_model(model, X, seq_len=args.seq)
    print('Saved model to data/models/transformer.pt')
    print('Preds shape:', preds.shape, 'Specs samples:', len(specs_list))


if __name__ == '__main__':
    main()


