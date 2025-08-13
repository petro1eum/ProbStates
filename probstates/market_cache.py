"""
Local caching helpers for market data and sentiment.

Stores:
- Prices: data/cache/prices/{ticker}_{start}_{end}_{column}.parquet
- News sentiment: data/cache/news/{symbol}_{days}.json

Requires yfinance for fetching; safe to import when unavailable (functions noop).
"""

from __future__ import annotations

import os
import json
from typing import Dict, Sequence
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def cache_root() -> str:
    return os.path.join('data', 'cache')


def fetch_prices_cached(tickers: Sequence[str], start: str, end: str, column: str = 'Close', force: bool = False, interval: str = '1d') -> Dict[str, pd.Series]:
    _ensure_dir(os.path.join(cache_root(), 'prices'))
    out: Dict[str, pd.Series] = {}
    for t in tickers:
        cache_fp = os.path.join(cache_root(), 'prices', f'{t}_{start}_{end}_{interval}_{column}.parquet')
        if (not force) and os.path.exists(cache_fp):
            try:
                s = pd.read_parquet(cache_fp)
                out[t] = s[column] if isinstance(s, pd.DataFrame) and column in s else s.squeeze()
                continue
            except Exception:
                pass
        if yf is None:
            continue
        try:
            df = yf.download(t, start=start, end=end, progress=False, auto_adjust=False, interval=interval)
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            col_used = None
            if column in df.columns:
                col_used = column
            elif column == 'Close' and 'Adj Close' in df.columns:
                col_used = 'Adj Close'
            if col_used is None:
                continue
            s = df[[col_used]].copy()
            # Normalize column name to requested one for consistency
            if col_used != column:
                s.rename(columns={col_used: column}, inplace=True)
            s.to_parquet(cache_fp)
            out[t] = s[column]
        except Exception:
            continue
    return out


def fetch_news_sentiment_cached(symbol: str, days: int, fetch_func, force: bool = False) -> pd.Series:
    _ensure_dir(os.path.join(cache_root(), 'news'))
    cache_fp = os.path.join(cache_root(), 'news', f'{symbol}_{days}.json')
    if (not force) and os.path.exists(cache_fp):
        try:
            with open(cache_fp, 'r') as f:
                data = json.load(f)
            s = pd.Series({pd.to_datetime(k).date(): v for k, v in data.items()})
            return s
        except Exception:
            pass
    dct = fetch_func(symbol, lookback_days=days) or {}
    try:
        with open(cache_fp, 'w') as f:
            json.dump({str(k): float(v) for k, v in dct.items()}, f)
    except Exception:
        pass
    return pd.Series(dct)


