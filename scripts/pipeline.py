#!/usr/bin/env python3
"""
Unified pipeline: cache → train → backtest → report

- Cache: prices for BTC and context tickers, news sentiment for BTC
- Train: transformer that learns adaptive weights/phases for ProbStates fusion
- Backtest: walk-forward (default) or single-period; optionally with transformer
- Report: prints metrics and saves CSV/JSON to data/reports/
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from probstates.market_cache import fetch_prices_cached, fetch_news_sentiment_cached
from probstates.markets import (
    momentum, rsi, sma, bollinger, macd, atr, realized_vol,
    fetch_yf_news_sentiment, indicator_to_prob, sentiment_to_phase,
    cross_asset_features, FeatureSpec, aggregate_specs, aggregate_specs_mc
)
from probstates.transformer import train_transformer, save_model, load_model, predict_with_model
from probstates import set_phase_or_mode
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def expand_with_proxies(tickers: list[str]) -> list[str]:
    """Augment list with proxy tickers for common indices when originals are unavailable."""
    proxies = {
        'DXY': ['UUP'],
        'DX-Y.NYB': ['UUP'],
        '^VIX': ['VIXY', 'VXX'],
    }
    out = list(tickers)
    for t, subs in proxies.items():
        if t in tickers:
            for s in subs:
                if s not in out:
                    out.append(s)
    return out


def _read_csv_series(fp: str, column: str = 'Close') -> Optional[pd.Series]:
    try:
        df = pd.read_csv(fp)
        # normalize columns to case-insensitive matching
        cols = {c.lower(): c for c in df.columns}
        date_col = cols.get('date') or cols.get('datetime')
        if not date_col:
            return None
        cand = None
        for key in [column.lower(), 'adj close', 'close']:
            if key in cols:
                cand = cols[key]
                break
        if not cand:
            return None
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        s = pd.Series(df[cand]).astype(float)
        s.name = 'Close'
        return s
    except Exception:
        return None


def load_prices_from_csv_dir(csv_dir: str, tickers: List[str], column: str = 'Close') -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for t in tickers:
        fp = os.path.join(csv_dir, f"{t}.csv")
        if os.path.exists(fp):
            s = _read_csv_series(fp, column=column)
            if s is not None and len(s) > 0:
                out[t] = s
    return out


def cache_step(tickers: list[str], start: str, end: str, interval: str = '1d', tiingo_token: Optional[str] = None) -> None:
    print(f"Caching {len(tickers)} tickers from {start} to {end} at interval={interval} ...")
    
    # If Tiingo token provided, try to collect fresh data first
    if tiingo_token:
        from probstates.data_sources import collect_to_csv
        ensure_dir('data/cache_tiingo')
        written = collect_to_csv(tickers, start, end, 'data/cache_tiingo', 
                               daily=(interval=='1d'), 
                               intraday_interval=interval if interval != '1d' else None,
                               tiingo_token=tiingo_token)
        print(f"Tiingo collected: {len(written)}/{len(tickers)} tickers")
    
    fetch_prices_cached(tickers, start, end, column='Close', force=False, interval=interval)
    fetch_news_sentiment_cached('BTC-USD', 365*3, fetch_yf_news_sentiment, force=False)


def robust_btc_df(start: str, end: str, synthetic: bool = False, interval: str = '1d', csv_dir: Optional[str] = None) -> pd.DataFrame:
    raw = yf.download('BTC-USD', start=start, end=end, progress=False, auto_adjust=False, interval=interval)
    df = None
    if isinstance(raw, pd.DataFrame) and not raw.empty:
        if isinstance(raw.columns, pd.MultiIndex):
            close_col = None
            for cand in [("Close","BTC-USD"),("Adj Close","BTC-USD")]:
                if cand in raw.columns:
                    close_col = cand; break
            if close_col is not None:
                out = pd.DataFrame({'Close': raw[close_col]})
                for name in ['High','Low']:
                    c = (name,'BTC-USD')
                    if c in raw.columns:
                        out[name] = raw[c]
                df = out.dropna(subset=['Close'])
        else:
            cols = list(raw.columns)
            if 'Close' in cols or 'Adj Close' in cols:
                cname = 'Close' if 'Close' in cols else 'Adj Close'
                out = pd.DataFrame({'Close': raw[cname]})
                if 'High' in cols and 'Low' in cols:
                    out['High'] = raw['High']; out['Low'] = raw['Low']
                df = out.dropna(subset=['Close'])
    if df is None or df.empty:
        cached = fetch_prices_cached(['BTC-USD'], start, end, column='Close', interval=interval)
        s = cached.get('BTC-USD')
        if s is not None and len(s) > 0:
            df = pd.DataFrame({'Close': s}).dropna()
        elif csv_dir:
            csv_map = load_prices_from_csv_dir(csv_dir, ['BTC-USD'])
            s2 = csv_map.get('BTC-USD')
            if s2 is not None and len(s2) > 0:
                df = pd.DataFrame({'Close': s2}).dropna()
        elif synthetic:
            # simple GBM synthetic
            rng = pd.date_range(start=start, end=end, freq='D')
            dt = 1.0/252.0; mu, sigma = 0.2, 0.6
            close = [20000.0]; rs = np.random.RandomState(42)
            for _ in range(len(rng)-1):
                z = rs.randn(); close.append(close[-1]*float(np.exp((mu-0.5*sigma*sigma)*dt+sigma*np.sqrt(dt)*z)))
            close = np.array(close)
            vol = np.clip(np.abs(rs.randn(len(rng)))*0.01, 0.002, 0.05)
            high = close*(1.0+vol); low = close*(1.0-vol)
            df = pd.DataFrame({'Close': close, 'High': high, 'Low': low}, index=rng)
        else:
            raise RuntimeError('No BTC data (network/cache). Use --synthetic to fallback.')
    return df


def attach_features(df: pd.DataFrame, start: str, end: str, ctx: list[str], interval: str = '1d', verbose: bool = False, csv_dir: Optional[str] = None, tiingo_cache_dir: Optional[str] = None) -> pd.DataFrame:
    df = df.copy()
    df['mom10'] = momentum(df['Close'].values, 10)
    df['rsi14'] = rsi(df['Close'].values, 14)
    df['sma20'] = sma(df['Close'].values, 20)
    df['sma50'] = sma(df['Close'].values, 50)
    mid, up, lo, pb = bollinger(df['Close'].values, 20)
    df['bb_pb'] = pb
    # Bollinger band width (scale-invariant trendiness/volatility)
    with np.errstate(invalid='ignore'):
        df['bb_width'] = (up - lo) / np.maximum(mid, 1e-12)
    macd_line, signal_line, hist = macd(df['Close'].values)
    df['macd_hist'] = hist
    if 'High' in df.columns and 'Low' in df.columns:
        df['atr14'] = atr(df['High'].values, df['Low'].values, df['Close'].values, 14)
    else:
        df['atr14'] = np.nan
    df['rv20'] = realized_vol(df['Close'].values, 20)
    # Scale-invariant engineered features
    df['ret1'] = np.log(df['Close']).diff()
    df['ret5'] = np.log(df['Close']).diff(5)
    df['ret20'] = np.log(df['Close']).diff(20)
    with np.errstate(divide='ignore', invalid='ignore'):
        df['atr_norm'] = df['atr14'] / df['Close']
        df['mom_norm'] = df['mom10'] / df['Close']
    # Center RSI around 0
    df['rsi_centered'] = (df['rsi14'] - 50.0) / 50.0
    # Volatility regime (zscore of rv20)
    rv = df['rv20'].values
    mu_rv = np.nanmean(rv); sd_rv = np.nanstd(rv) + 1e-12
    df['rv_z'] = (df['rv20'] - mu_rv) / sd_rv
    # news
    s_series = fetch_news_sentiment_cached('BTC-USD', 365*3, fetch_yf_news_sentiment)
    if not s_series.empty:
        s_series.index = pd.to_datetime(s_series.index)
        daily_s = s_series.resample('D').mean().ffill()
        df['news_sent'] = daily_s.reindex(df.index, method='ffill').fillna(0.0)
    else:
        df['news_sent'] = 0.0
    # context
    if ctx:
        ctx_raw = fetch_prices_cached(ctx, start, end, column='Close', interval=interval)
        # keep only non-empty
        ctx_raw = {k: v for k, v in ctx_raw.items() if v is not None and len(v) > 0}
        
        # fallback to Tiingo cache for missing
        if tiingo_cache_dir:
            missing = [t for t in ctx if t not in ctx_raw]
            for ticker in missing:
                tiingo_path = os.path.join(tiingo_cache_dir, f"{ticker}.csv")
                if os.path.exists(tiingo_path):
                    try:
                        ctx_df = pd.read_csv(tiingo_path, parse_dates=['Date'], index_col='Date')
                        if 'Close' in ctx_df.columns and not ctx_df.empty:
                            ctx_raw[ticker] = ctx_df['Close']
                    except Exception:
                        pass
        
        # fallback to CSV for still missing
        if csv_dir:
            missing = [t for t in ctx if t not in ctx_raw]
            if missing:
                csv_loaded = load_prices_from_csv_dir(csv_dir, missing)
                ctx_raw.update(csv_loaded)
        ctx_map = {k: (v.values if isinstance(v, pd.Series) else v) for k, v in ctx_raw.items()}
        if ctx_map:
            feats = cross_asset_features(df['Close'].values, ctx_map, windows=(20,60))
            for k, arr in feats.items():
                df[k] = arr
        if verbose:
            print(f"Context tickers ok: {len(ctx_map)}/{len(ctx)} → {sorted(list(ctx_map.keys()))}")
    # dropna on core
    core_cols = ['Close','mom10','rsi14','sma20','sma50','bb_pb','macd_hist','rv20']
    if ('High' in df.columns) and ('Low' in df.columns) and ('atr14' in df.columns):
        core_cols.append('atr14')
    df.dropna(subset=[c for c in core_cols if c in df.columns], inplace=True)
    return df


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    base = [
        'mom10','rsi14','sma20','sma50','bb_pb','bb_width','macd_hist','atr14','rv20',
        'ret1','ret5','ret20','atr_norm','mom_norm','rsi_centered','rv_z'
    ]
    cols = [c for c in base if c in df.columns]
    ctx_cols = sorted([c for c in df.columns if isinstance(c, str) and (c.startswith('corr20_') or c.startswith('mom20_') or c.startswith('risk_'))])
    cols.extend(ctx_cols)
    return df[cols].values.astype(float)


def compute_agg_p(df: pd.DataFrame, mode: str) -> pd.Series:
    set_phase_or_mode(mode)
    out = []
    for _, row in df.iterrows():
        specs: list[FeatureSpec] = []
        mom = float(row['mom10']); p_mom = float(indicator_to_prob(np.array([mom]))[-1]); phi_mom = 0.0 if mom>=0 else np.pi
        specs.append(FeatureSpec('momentum_10', p_mom, phi_mom, 2))
        rsi_v = float(row['rsi14']); p_rsi = float(indicator_to_prob(np.array([rsi_v]), center=50.0, k=1.2)[-1]); phi_rsi = 0 if rsi_v>=50 else np.pi
        specs.append(FeatureSpec('rsi_14', p_rsi, phi_rsi, 1))
        cross = float(row['sma20'] - row['sma50']); p_cross = float(indicator_to_prob(np.array([cross]))[-1]); phi_cross = 0 if cross>=0 else np.pi
        specs.append(FeatureSpec('sma20_vs_50', p_cross, phi_cross, 2))
        pb = float(row['bb_pb']); p_pb = float(indicator_to_prob(np.array([pb]), center=0.5, k=1.2)[-1]); phi_pb = 0 if pb>=0.5 else np.pi
        specs.append(FeatureSpec('bb_pb', p_pb, phi_pb, 1))
        mh = float(row['macd_hist']); p_mh = float(indicator_to_prob(np.array([mh]))[-1]); phi_mh = 0 if mh>=0 else np.pi
        specs.append(FeatureSpec('macd_hist', p_mh, phi_mh, 1))
        rv = float(row['rv20']); conf = float(np.clip(1.0 - float(np.tanh(abs(rv))), 0.0, 1.0))
        specs.append(FeatureSpec('rv_conf', conf, 0.0 if conf>=0.5 else np.pi, 1))
        if 'news_sent' in row:
            s = float(row['news_sent']); p_news = 0.5 + 0.5*abs(s); phi_news = sentiment_to_phase(s)
            specs.append(FeatureSpec('news', p_news, phi_news, 2))
        for key, val in row.items():
            if isinstance(key, str) and key.startswith('corr20_'):
                t = key.split('corr20_')[-1]; mom_key = f'mom20_{t}'
                if mom_key in row:
                    corr = float(val); momv = float(row[mom_key])
                    if not (np.isnan(corr) or np.isnan(momv)):
                        score = corr * momv; p = float(indicator_to_prob(np.array([score]), k=1.0)[-1]); phi = 0 if score>=0 else np.pi
                        specs.append(FeatureSpec(f'ctx_{t}', p, phi, 1))
            if isinstance(key, str) and key.startswith('risk_'):
                r = float(val)
                if not np.isnan(r):
                    p_r = float(indicator_to_prob(np.array([r]), k=1.0)[-1]); phi_r = 0 if r>=0 else np.pi
                    specs.append(FeatureSpec(key, p_r, phi_r, 1))
        out.append(aggregate_specs(specs, mode=mode).probability)
    return pd.Series(out, index=df.index)


def metrics(series: pd.Series) -> dict:
    x = series.dropna().values
    if x.size == 0:
        return dict(CAGR=0.0, Sharpe=0.0, MDD=0.0)
    ann = np.prod(1.0 + x) ** (252/len(x)) - 1.0
    vol = np.std(x) * np.sqrt(252)
    sharpe = ann / (vol + 1e-12)
    curve = (1.0 + x).cumprod()
    peak = np.maximum.accumulate(curve)
    mdd = float(np.max((peak - curve)/peak))
    return dict(CAGR=float(ann), Sharpe=float(sharpe), MDD=float(mdd))


def walk_forward(df: pd.DataFrame, fee: float, mode: str) -> tuple[pd.DataFrame, pd.Series]:
    years = sorted(set(df.index.year))
    results = []; all_strat = []
    for i in range(1, len(years)):
        train = df[df.index.year == years[i-1]]
        test = df[df.index.year == years[i]]
        if len(train) < 100 or len(test) < 50:
            continue
        agg_train = compute_agg_p(train, mode=mode)
        ret_tr = train['Close'].pct_change().fillna(0.0)
        best = (-1e9, 0.55, 0.45)
        for buy in np.linspace(0.52, 0.65, 14):
            for sell in np.linspace(0.35, 0.48, 14):
                pos = np.where(agg_train.values >= buy, 1.0, np.where(agg_train.values <= sell, 0.0, np.nan))
                pos = pd.Series(pos, index=agg_train.index).ffill().fillna(0.0)
                trade = pos.diff().abs().fillna(0.0)
                strat = pos.shift(1).fillna(0.0) * ret_tr - fee * trade
                score = metrics(strat)['Sharpe']
                if score > best[0]:
                    best = (score, buy, sell)
        agg_test = compute_agg_p(test, mode=mode)
        ret_ts = test['Close'].pct_change().fillna(0.0)
        pos = np.where(agg_test.values >= best[1], 1.0, np.where(agg_test.values <= best[2], 0.0, np.nan))
        pos = pd.Series(pos, index=agg_test.index).ffill().fillna(0.0)
        trade = pos.diff().abs().fillna(0.0)
        strat = pos.shift(1).fillna(0.0) * ret_ts - fee * trade
        res = dict(year=int(years[i]), buy=float(best[1]), sell=float(best[2]))
        res.update({f'test_{k}': v for k, v in metrics(strat).items()})
        res['train_sharpe'] = float(best[0])
        results.append(res); all_strat.append(strat)
    joined = pd.concat(all_strat).sort_index() if all_strat else pd.Series(dtype=float)
    return pd.DataFrame(results), joined


def backtest_with_transformer(df: pd.DataFrame, model_path: str, seq_len: int, fee: float, mode: str, kappa_prog: float | None = None, mc: int = 128) -> tuple[pd.DataFrame, pd.Series]:
    model = load_model(model_path)
    set_phase_or_mode(mode)
    X = build_feature_matrix(df)
    probs = []
    for i in range(len(df)):
        if i + 1 < seq_len:
            probs.append(np.nan); continue
        specs = model.build_specs(X[i])
        # Handle NaN predictions
        for spec in specs:
            if np.isnan(spec.prob):
                spec.prob = 0.5  # neutral probability
            if np.isnan(spec.phase):
                spec.phase = 0.0  # neutral phase
        # Prefer MC aggregation if uncertainty params are present or κ_prog specified
        if any((s.alpha > 0 and s.beta > 0) or (s.kappa_vm > 0) for s in specs) or (kappa_prog is not None):
            agg_state = aggregate_specs_mc(specs, samples=mc, mode=mode, kappa_prog=kappa_prog)
        else:
            agg_state = aggregate_specs(specs, mode=mode)
        probs.append(agg_state.probability)
    out = df.copy(); out['agg_p'] = pd.Series(probs, index=out.index).ffill().fillna(0.5)
    ret = out['Close'].pct_change().fillna(0.0)
    pos = (out['agg_p'] >= 0.55).astype(float).ffill().fillna(0.0)
    trade = pos.diff().abs().fillna(0.0)
    strat = pos.shift(1).fillna(0.0) * ret - fee * trade
    return out, strat


def backtest_with_model(model, df: pd.DataFrame, seq_len: int, fee: float, mode: str, kappa_prog: float | None = None, mc: int = 128) -> tuple[pd.DataFrame, pd.Series]:
    set_phase_or_mode(mode)
    X = build_feature_matrix(df)
    # Use model's batch predictor to avoid per-step forward calls
    _, specs_list_all = predict_with_model(model, X, seq_len=seq_len)
    probs: list[float] = [np.nan] * min(seq_len - 1, len(df))
    for specs in specs_list_all:
        for spec in specs:
            if np.isnan(spec.prob):
                spec.prob = 0.5
            if np.isnan(spec.phase):
                spec.phase = 0.0
        if any((s.alpha > 0 and s.beta > 0) or (s.kappa_vm > 0) for s in specs) or (kappa_prog is not None):
            agg_state = aggregate_specs_mc(specs, samples=mc, mode=mode, kappa_prog=kappa_prog)
        else:
            agg_state = aggregate_specs(specs, mode=mode)
        probs.append(float(agg_state.probability))
    out = df.copy(); out['agg_p'] = pd.Series(probs, index=out.index).ffill().fillna(0.5)
    ret = out['Close'].pct_change().fillna(0.0)
    pos = (out['agg_p'] >= 0.55).astype(float).ffill().fillna(0.0)
    trade = pos.diff().abs().fillna(0.0)
    strat = pos.shift(1).fillna(0.0) * ret - fee * trade
    return out, strat


def walk_forward_transformer(
    df: pd.DataFrame,
    seq_len: int,
    fee: float,
    mode: str,
    fusion_train: bool,
    l1_weight: float,
    d_model: int,
    nhead: int,
    layers: int,
    batch: int,
    epochs: int,
    mc: int,
    kappa_prog: float | None,
    kappa_points: int,
    thr_points: int,
) -> tuple[pd.DataFrame, pd.Series]:
    years = sorted(list({int(d.year) for d in df.index}))
    results: list[dict] = []
    all_strat: list[pd.Series] = []
    for y in years:
        tr_mask = df.index.year < y
        ts_mask = df.index.year == y
        df_tr = df.loc[tr_mask]
        df_ts = df.loc[ts_mask]
        if df_tr.empty or df_ts.empty:
            continue
        X_tr = build_feature_matrix(df_tr)
        y_tr = df_tr['Close'].pct_change().shift(-1).fillna(0.0).values
        if X_tr.shape[0] <= seq_len:
            continue
        print(f"WF year {y}: train={df_tr.index.min().date()}..{df_tr.index.max().date()} test={df_ts.index.min().date()}..{df_ts.index.max().date()} (n_tr={X_tr.shape[0]})")
        model = train_transformer(
            X_tr, y_tr,
            seq_len=seq_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=layers,
            batch_size=batch,
            epochs=epochs,
            mode=mode,
            fusion_train=fusion_train,
            l1_weight=l1_weight,
        )
        # Calibrate kappa on train window if not provided
        if kappa_prog is None:
            k_grid = np.linspace(-0.9, 0.9, max(3, int(kappa_points)))
        else:
            k_grid = np.array([kappa_prog])
        best_k = None
        best_thr = (0.55, 0.45)
        best_score = -1e18
        df_tr_out, _ = backtest_with_model(model, df_tr, seq_len, fee=0.0, mode=mode, kappa_prog=None, mc=mc)
        agg = df_tr_out['agg_p']
        ret_tr = df_tr_out['Close'].pct_change().fillna(0.0)
        for k in k_grid:
            df_k, _ = backtest_with_model(model, df_tr, seq_len, fee=0.0, mode=mode, kappa_prog=float(k), mc=mc)
            agg_k = df_k['agg_p']
            for buy in np.linspace(0.52, 0.65, max(3, int(thr_points))):
                for sell in np.linspace(0.35, 0.48, max(3, int(thr_points))):
                    pos = np.where(agg_k.values >= buy, 1.0, np.where(agg_k.values <= sell, 0.0, np.nan))
                    pos = pd.Series(pos, index=agg_k.index).ffill().fillna(0.0)
                    trade = pos.diff().abs().fillna(0.0)
                    strat = pos.shift(1).fillna(0.0) * ret_tr - fee * trade
                    score = metrics(strat)['Sharpe']
                    if score > best_score:
                        best_score = score
                        best_k = float(k)
                        best_thr = (float(buy), float(sell))
        print(f"  tuned kappa={best_k}, buy={best_thr[0]:.3f}, sell={best_thr[1]:.3f}, train Sharpe={best_score:.3f}")
        # Evaluate on test with best kappa and thresholds
        df_ts_out, _ = backtest_with_model(model, df_ts, seq_len, fee=0.0, mode=mode, kappa_prog=best_k, mc=mc)
        agg_ts = df_ts_out['agg_p']
        ret_ts = df_ts_out['Close'].pct_change().fillna(0.0)
        pos = np.where(agg_ts.values >= best_thr[0], 1.0, np.where(agg_ts.values <= best_thr[1], 0.0, np.nan))
        pos = pd.Series(pos, index=agg_ts.index).ffill().fillna(0.0)
        trade = pos.diff().abs().fillna(0.0)
        strat = pos.shift(1).fillna(0.0) * ret_ts - fee * trade
        res = dict(year=int(y), kappa=float(best_k) if best_k is not None else None, buy=float(best_thr[0]), sell=float(best_thr[1]))
        res.update(metrics(strat))
        results.append(res)
        all_strat.append(strat)
    joined = pd.concat(all_strat).sort_index() if all_strat else pd.Series(dtype=float)
    return pd.DataFrame(results), joined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2014-01-01')
    ap.add_argument('--end', default='2025-01-01')
    ap.add_argument('--ctx', default='DX-Y.NYB,^VIX,SPY,GLD,ETH-USD')
    ap.add_argument('--ctx_full', action='store_true', help='append a larger preset of macro+crypto tickers')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--seq', type=int, default=64)
    ap.add_argument('--d_model', type=int, default=128)
    ap.add_argument('--nhead', type=int, default=4)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--mode', default='weight', choices=['weight','norm','quant','opt'])
    ap.add_argument('--fee', type=float, default=0.001)
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--no_train', action='store_true')
    ap.add_argument('--no_backtest', action='store_true')
    ap.add_argument('--use_transformer', action='store_true')
    ap.add_argument('--model_path', default='data/models/transformer.pt')
    ap.add_argument('--interval', default='1d', choices=['1d','1h','30m','15m'], help='price resolution')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    ap.add_argument('--val', type=float, default=0.2, help='validation ratio')
    ap.add_argument('--patience', type=int, default=5, help='early stopping patience')
    # already added above
    # already added above
    ap.add_argument('--kappa_prog', type=float, default=None, help='programmable fusion target E[cos Δφ]=κ ∈ [-1,1]')
    ap.add_argument('--mc', type=int, default=128, help='MC samples for aggregation')
    ap.add_argument('--fusion_train', action='store_true', help='optimize transformer on E[p_agg] via fusion loss')
    ap.add_argument('--l1_weight', type=float, default=1e-4, help='L1 penalty on feature weights in fusion head')
    ap.add_argument('--csv_dir', default=None, help='optional directory with per-ticker CSVs (e.g., BTC-USD.csv)')
    ap.add_argument('--tiingo_token', default=None, help='Tiingo API token for premium data')
    # explicit chronological splits
    ap.add_argument('--train_start', default=None)
    ap.add_argument('--train_end', default=None)
    ap.add_argument('--val_start', default=None)
    ap.add_argument('--val_end', default=None)
    ap.add_argument('--test_start', default=None)
    ap.add_argument('--test_end', default=None)
    # Risk management
    ap.add_argument('--risk', default='none', choices=['none', 'vt', 'vt_atr'], help='risk control mode')
    ap.add_argument('--vt_target', type=float, default=0.2, help='annual vol target for vt')
    ap.add_argument('--atr_k', type=float, default=3.0, help='ATR multiple for trailing stop (vt_atr)')
    ap.add_argument('--slip', type=float, default=0.0, help='slippage per turnover unit')
    args = ap.parse_args()

    tickers = ['BTC-USD'] + [t.strip() for t in (args.ctx or '').split(',') if t.strip()]
    tickers = expand_with_proxies(tickers)
    if args.ctx_full:
        preset = [
            '^GSPC','^IXIC','^DJI','^RUT','^VIX','DX-Y.NYB','DXY','EURUSD=X','JPY=X',
            'SPY','QQQ','TLT','IEF','HYG','USO','GLD','SLV',
            'ETH-USD','BNB-USD','SOL-USD','ADA-USD','XRP-USD','LTC-USD','DOGE-USD','AVAX-USD','MATIC-USD'
        ]
        tickers += [t for t in preset if t not in tickers]

    # Validate intraday intervals vs span
    def _max_days(iv: str) -> int:
        return {'1h': 730, '30m': 60, '15m': 60}.get(iv, 1000000)
    start_dt = datetime.fromisoformat(args.start)
    end_dt = datetime.fromisoformat(args.end)
    span_days = (end_dt - start_dt).days
    interval = args.interval
    if interval != '1d' and span_days > _max_days(interval):
        print(f"Requested interval={interval} exceeds provider limit for span {span_days}d. Falling back to 1d.")
        interval = '1d'

    print('Step 1/4: Cache')
    cache_step(tickers, args.start, args.end, interval=interval, tiingo_token=args.tiingo_token)

    print('Step 2/4: Build dataset')
    df = robust_btc_df(args.start, args.end, synthetic=args.synthetic, interval=interval, csv_dir=args.csv_dir)
    df = attach_features(
        df, args.start, args.end,
        ctx=[t for t in tickers if t != 'BTC-USD'],
        interval=interval,
        verbose=args.verbose,
        csv_dir=args.csv_dir,
        tiingo_cache_dir='data/cache_tiingo' if args.tiingo_token else None,
    )
    if df.empty:
        raise RuntimeError('Empty dataframe after feature computation')
    X = build_feature_matrix(df)
    y = df['Close'].pct_change().shift(-1).fillna(0.0).values
    if args.verbose:
        ctx_cols = [c for c in df.columns if isinstance(c, str) and (c.startswith('corr20_') or c.startswith('mom20_') or c.startswith('risk_'))]
        print(f"Dataset built: rows={X.shape[0]}, features={X.shape[1]} (ctx={len(ctx_cols)}), start={df.index.min().date()}, end={df.index.max().date()}")

    reports_dir = os.path.join('data','reports', datetime.now().strftime('%Y%m%d_%H%M%S'))
    ensure_dir(reports_dir)

    # Prepare splits
    def _slice(df_local: pd.DataFrame, s: Optional[str], e: Optional[str]) -> pd.DataFrame:
        if s is None and e is None:
            return df_local
        mask = pd.Series(True, index=df_local.index)
        if s is not None:
            mask &= (df_local.index >= pd.to_datetime(s))
        if e is not None:
            mask &= (df_local.index <= pd.to_datetime(e))
        return df_local.loc[mask]

    df_train = _slice(df, args.train_start, args.train_end)
    df_val = _slice(df, args.val_start, args.val_end)
    df_test = _slice(df, args.test_start, args.test_end)

    if args.verbose and (args.train_start or args.val_start or args.test_start):
        print(f"Splits: train=[{df_train.index.min().date() if not df_train.empty else 'NA'} .. {df_train.index.max().date() if not df_train.empty else 'NA'}]",
              f"val=[{df_val.index.min().date() if not df_val.empty else 'NA'} .. {df_val.index.max().date() if not df_val.empty else 'NA'}]",
              f"test=[{df_test.index.min().date() if not df_test.empty else 'NA'} .. {df_test.index.max().date() if not df_test.empty else 'NA'}]")

    if not args.no_train:
        print('Step 3/4: Train transformer')
        X_tr = build_feature_matrix(df_train) if not df_train.empty else X
        y_tr = df_train['Close'].pct_change().shift(-1).fillna(0.0).values if not df_train.empty else y
        print(f'Dataset: {X_tr.shape[0]} rows, {X_tr.shape[1]} features; training for {args.epochs} epochs ...')
        model = train_transformer(
            X_tr, y_tr,
            seq_len=args.seq,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.layers,
            batch_size=args.batch,
            epochs=args.epochs,
            mode=args.mode,
            weight_decay=args.wd,
            val_ratio=args.val,
            patience=args.patience,
            fusion_train=args.fusion_train,
            l1_weight=args.l1_weight,
        )
        save_model(model, args.model_path)
        print('Saved model to', args.model_path)

    if not args.no_backtest:
        print('Step 4/4: Backtest and report')
        eval_df = df_test if (args.test_start or args.test_end) else df
        if args.use_transformer:
            df2, _ = backtest_with_transformer(eval_df, args.model_path, seq_len=args.seq, fee=args.fee, mode=args.mode, kappa_prog=args.kappa_prog, mc=args.mc)
        else:
            agg = compute_agg_p(eval_df, mode=args.mode)
            df2 = eval_df.copy(); df2['agg_p'] = agg
        # Apply risk management
        ret = df2['Close'].pct_change().fillna(0.0)
        pos = (df2['agg_p'] >= 0.55).astype(float).ffill().fillna(0.0)
        if args.risk in ('vt','vt_atr'):
            rv = df2['rv20'].ffill().fillna(0.0)
            scale = np.clip(args.vt_target / (rv + 1e-6), 0.0, 1.5)
            pos = (pos * scale).clip(0.0, 1.0)
        if args.risk == 'vt_atr' and 'atr14' in df2.columns:
            peak = np.maximum.accumulate(df2['Close'].values)
            stop = peak - args.atr_k * df2['atr14'].ffill().fillna(0.0).values
            active = (df2['Close'].values >= stop).astype(float)
            pos = pos * active
        trade = pd.Series(pos, index=df2.index).diff().abs().fillna(0.0)
        strat = pd.Series(pos, index=df2.index).shift(1).fillna(0.0) * ret - args.fee * trade - args.slip * trade
        m = metrics(strat)
        # Walk-forward with retrain per year using transformer if enabled
        if args.use_transformer:
            table, joined = walk_forward_transformer(
                df,
                seq_len=args.seq,
                fee=args.fee,
                mode=args.mode,
                fusion_train=args.fusion_train,
                l1_weight=args.l1_weight,
                d_model=args.d_model,
                nhead=args.nhead,
                layers=args.layers,
                batch=args.batch,
                epochs=args.epochs,
                mc=args.mc,
                kappa_prog=args.kappa_prog,
                kappa_points=7,
                thr_points=6,
            )
        else:
            table, joined = walk_forward(df, fee=args.fee, mode=args.mode)
        print('Strategy:', {k: round(v,3) for k,v in m.items()})
        if not joined.empty:
            print('WF total:', {k: round(v,3) for k, v in metrics(joined).items()})
        # Save report
        table.to_csv(os.path.join(reports_dir, 'walk_forward.csv'), index=False)
        with open(os.path.join(reports_dir, 'summary.json'), 'w') as f:
            json.dump({'strategy': m, 'wf_total': (metrics(joined) if not joined.empty else {})}, f, indent=2)
        # Charts: price + position, equity curves, drawdown
        eq_strat = (1.0 + strat.fillna(0.0)).cumprod()
        eq_bh = (1.0 + df2['Close'].pct_change().fillna(0.0)).cumprod()
        plt.figure(figsize=(10,6))
        ax1 = plt.gca()
        ax1.plot(df2.index, df2['Close'], color='tab:blue', label='BTC Close')
        ax1.set_ylabel('Price', color='tab:blue')
        ax2 = ax1.twinx()
        ax2.plot(df2.index, df2.get('agg_p', pd.Series(index=df2.index, dtype=float)), color='tab:orange', alpha=0.6, label='agg_p')
        ax2.set_ylabel('agg_p', color='tab:orange')
        ax1.set_title('BTC & Aggregated Probability')
        plt.tight_layout(); plt.savefig(os.path.join(reports_dir, 'price_prob.png')); plt.close()

        plt.figure(figsize=(10,6))
        plt.plot(eq_strat.index, eq_strat.values, label='Strategy')
        plt.plot(eq_bh.index, eq_bh.values, label='Buy&Hold')
        plt.legend(); plt.title('Equity Curves'); plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, 'equity.png')); plt.close()

        # drawdown chart
        curve = eq_strat.values
        peak = np.maximum.accumulate(curve)
        dd = (peak - curve) / np.maximum(peak, 1e-12)
        plt.figure(figsize=(10,4))
        plt.fill_between(eq_strat.index, dd, color='tab:red', alpha=0.4)
        plt.title('Drawdown (Strategy)')
        plt.tight_layout(); plt.savefig(os.path.join(reports_dir, 'drawdown.png')); plt.close()

        # Simple HTML report
        html = f"""
        <html><head><meta charset='utf-8'><title>ProbStates Report</title></head>
        <body>
        <h2>Summary</h2>
        <pre>{json.dumps(m, indent=2)}</pre>
        <h2>Price & Agg Probability</h2>
        <img src='price_prob.png' width='900'/>
        <h2>Equity Curves</h2>
        <img src='equity.png' width='900'/>
        <h2>Walk-Forward</h2>
        <pre>{table.to_string(index=False)}</pre>
        <h2>Drawdown</h2>
        <img src='drawdown.png' width='900'/>
        </body></html>
        """
        with open(os.path.join(reports_dir, 'report.html'), 'w') as f:
            f.write(html)
        print('Saved report to', reports_dir)


if __name__ == '__main__':
    main()


