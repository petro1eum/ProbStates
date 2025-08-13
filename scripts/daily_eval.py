#!/usr/bin/env python3
"""
Day-by-day decision evaluation for BTC using ProbStates.

For each evaluation date d in [start, end]:
- Use only data up to and including d to compute aggregated probability agg_p
- Produce decision (buy/sell/hold) by thresholds
- Compare with realized next-day return and compute strategy PnL with fees

Outputs:
- CSV log with per-day decisions and realized metrics
- JSON summary with accuracy, Sharpe, CAGR, MDD
"""

from __future__ import annotations

import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from probstates import set_phase_or_mode
from probstates.market_cache import fetch_prices_cached, fetch_news_sentiment_cached
from probstates.markets import FeatureSpec, aggregate_specs_mc
try:
    from scripts.pipeline import (
        attach_features,
        robust_btc_df,
        build_feature_matrix,
        compute_agg_p,
    )
except Exception:
    from pipeline import (
        attach_features,
        robust_btc_df,
        build_feature_matrix,
        compute_agg_p,
    )
from probstates.transformer import load_model, predict_with_model


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def metrics(series: pd.Series) -> dict:
    x = series.dropna().values
    if x.size == 0:
        return dict(CAGR=0.0, Sharpe=0.0, MDD=0.0)
    ann = np.prod(1.0 + x) ** (252 / len(x)) - 1.0
    vol = np.std(x) * np.sqrt(252)
    sharpe = ann / (vol + 1e-12)
    curve = (1.0 + x).cumprod()
    peak = np.maximum.accumulate(curve)
    mdd = float(np.max((peak - curve) / np.maximum(peak, 1e-12)))
    return dict(CAGR=float(ann), Sharpe=float(sharpe), MDD=float(mdd))


def last_specs_agg_p(model, df_hist: pd.DataFrame, seq_len: int, mode: str, kappa_prog: Optional[float], mc: int) -> float:
    """Compute agg_p on the last date using transformer model and ProbStates aggregation.
    Uses only df_hist up to the date.
    """
    set_phase_or_mode(mode)
    X = build_feature_matrix(df_hist)
    # Align feature dimension with the trained model if mismatch
    try:
        nf = int(getattr(model.fusion_cfg, 'num_features', X.shape[1]))
    except Exception:
        nf = X.shape[1]
    if X.shape[1] != nf:
        import numpy as np  # local import safe
        if X.shape[1] < nf:
            pad = np.zeros((X.shape[0], nf - X.shape[1]), dtype=X.dtype)
            X = np.concatenate([X, pad], axis=1)
        else:
            X = X[:, :nf]
    # Batch prediction for entire history; we only take the last step's specs
    _, specs_list = predict_with_model(model, X, seq_len=seq_len)
    if len(specs_list) == 0:
        return float('nan')
    specs = specs_list[-1]
    # Sanitize NaNs
    for s in specs:
        if np.isnan(s.prob):
            s.prob = 0.5
        if np.isnan(s.phase):
            s.phase = 0.0
    agg_state = aggregate_specs_mc(specs, samples=mc, mode=mode, kappa_prog=kappa_prog)
    return float(agg_state.probability)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', required=True, help='evaluation start date (YYYY-MM-DD)')
    ap.add_argument('--end', required=True, help='evaluation end date (YYYY-MM-DD)')
    ap.add_argument('--ctx', default='SPY,QQQ,TLT,IEF,GLD,UUP,VIXY,XLK,XLF,XLV,USO,EEM')
    ap.add_argument('--interval', default='1d', choices=['1d', '1h', '30m', '15m'])
    ap.add_argument('--mode', default='weight', choices=['weight', 'norm', 'quant', 'opt'])
    ap.add_argument('--buy', type=float, default=0.55)
    ap.add_argument('--sell', type=float, default=0.45)
    ap.add_argument('--fee', type=float, default=0.001)
    ap.add_argument('--slip', type=float, default=0.0)
    ap.add_argument('--use_transformer', action='store_true')
    ap.add_argument('--model_path', default='data/models/transformer.pt')
    ap.add_argument('--seq', type=int, default=64)
    ap.add_argument('--kappa_prog', type=float, default=None)
    ap.add_argument('--mc', type=int, default=128)
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--csv_dir', default=None, help='optional directory with cached CSVs (e.g., data/cache_tiingo)')
    ap.add_argument('--tiingo_cache_dir', default=None, help='directory with Tiingo CSV cache (e.g., data/cache_tiingo)')
    # Optional train/test split inside [start, end] for calibration
    ap.add_argument('--train_start', default=None)
    ap.add_argument('--train_end', default=None)
    ap.add_argument('--test_start', default=None)
    ap.add_argument('--test_end', default=None)
    # Risk control
    ap.add_argument('--risk', default='none', choices=['none','vt','vt_atr'])
    ap.add_argument('--vt_target', type=float, default=0.2)
    ap.add_argument('--atr_k', type=float, default=3.0)
    # Grid search for calibration (on train)
    ap.add_argument('--grid_modes', default='weight,norm,opt')
    ap.add_argument('--grid_kappa', default='-0.9,-0.6,-0.3,0.0,0.3')
    ap.add_argument('--grid_buy', default='0.55:0.65:6')
    ap.add_argument('--grid_sell', default='0.35:0.48:6')
    args = ap.parse_args()

    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end)
    ctx = [t.strip() for t in (args.ctx or '').split(',') if t.strip()]

    # Build dataset once with extended history to satisfy indicator lookbacks
    def _max_days(iv: str) -> int:
        return {'1h': 730, '30m': 60, '15m': 60}.get(iv, 1000000)
    buffer_days = 400  # covers SMA50, rv20, MACD26, etc.
    ext_start = (start - pd.Timedelta(days=min(buffer_days, _max_days(args.interval) - 1))).strftime('%Y-%m-%d')
    df_raw = robust_btc_df(ext_start, args.end, synthetic=False, interval=args.interval, csv_dir=args.csv_dir)
    df_full = attach_features(
        df_raw, ext_start, args.end,
        ctx=ctx,
        interval=args.interval,
        verbose=False,
        csv_dir=args.csv_dir,
        tiingo_cache_dir=(args.tiingo_cache_dir or ('data/cache_tiingo' if os.path.isdir('data/cache_tiingo') else None)),
    )
    # Normalize timezone for intraday (yfinance often returns UTC-aware index)
    try:
        idx = df_full.index
        # If tz-aware, convert to UTC and drop tz for consistent comparisons
        if getattr(idx, 'tz', None) is not None:
            df_full = df_full.tz_convert('UTC')
            df_full.index = df_full.index.tz_localize(None)
    except Exception:
        pass
    # Evaluation slice (decisions only on [start, end], but history uses df_full)
    df = df_full.loc[(df_full.index >= start) & (df_full.index <= end)].copy()
    if df.empty:
        raise RuntimeError('No data in the requested evaluation window')

    model = None
    if args.use_transformer:
        model = load_model(args.model_path)

    # Prepare loop
    dates = list(df.index)
    kappa_prog = None if args.kappa_prog is None else float(np.clip(args.kappa_prog, -1.0, 1.0))
    log_rows: List[Dict] = []
    pos_prev: float = 0.0
    strat_returns: List[Tuple[pd.Timestamp, float]] = []

    def _apply_risk(df_loc: pd.DataFrame, pos: pd.Series) -> pd.Series:
        if args.risk in ('vt','vt_atr'):
            rv = df_loc['rv20'].ffill().fillna(0.0)
            scale = np.clip(args.vt_target / (rv + 1e-6), 0.0, 1.5)
            pos = (pos * scale).clip(0.0, 1.0)
        if args.risk == 'vt_atr' and 'atr14' in df_loc.columns:
            peak = np.maximum.accumulate(df_loc['Close'].values)
            stop = peak - args.atr_k * df_loc['atr14'].ffill().fillna(0.0).values
            active = (df_loc['Close'].values >= stop).astype(float)
            pos = pos * active
        return pos

    def _evaluate_series(agg: pd.Series, df_slice: pd.DataFrame, buy_thr: float, sell_thr: float, thr_gamma: float = 0.0) -> tuple[pd.Series, dict]:
        ret = df_slice['Close'].pct_change().fillna(0.0)
        # Adaptive thresholds by volatility regime
        if thr_gamma != 0.0:
            rv = df_slice['rv20'].ffill().fillna(0.0).values
            rv_mu = float(np.nanmean(rv) or 1e-6)
            scale = 1.0 + thr_gamma * (rv / (rv_mu + 1e-6))
            buy_arr = np.clip(buy_thr * scale, 0.50, 0.70)
            sell_arr = np.clip(sell_thr * (2.0 - scale), 0.30, 0.50)
            vals = agg.values
            pos = np.where(vals >= buy_arr, 1.0, np.where(vals <= sell_arr, 0.0, np.nan))
        else:
            pos = np.where(agg.values >= buy_thr, 1.0, np.where(agg.values <= sell_thr, 0.0, np.nan))
        pos = pd.Series(pos, index=agg.index).ffill().fillna(0.0)
        pos = _apply_risk(df_slice, pos)
        trade = pos.diff().abs().fillna(0.0)
        strat = pos.shift(1).fillna(0.0) * ret - args.fee * trade - args.slip * trade
        return strat, metrics(strat)

    def _agg_series(df_slice: pd.DataFrame, mode_use: str, kappa_use: Optional[float]) -> pd.Series:
        # Compute aggregated probability series across df_slice using transformer or baseline
        if args.use_transformer:
            # Use model batch predictor for efficiency
            Xs = build_feature_matrix(df_slice)
            _, specs_list = predict_with_model(model, Xs, seq_len=args.seq)
            vals: list[float] = [np.nan] * min(args.seq - 1, len(df_slice))
            for specs in specs_list:
                for s in specs:
                    if np.isnan(s.prob):
                        s.prob = 0.5
                    if np.isnan(s.phase):
                        s.phase = 0.0
                agg_state = aggregate_specs_mc(specs, samples=args.mc, mode=mode_use, kappa_prog=kappa_use)
                vals.append(float(agg_state.probability))
            return pd.Series(vals, index=df_slice.index).ffill().fillna(0.5)
        else:
            # fallback simple per-row aggregation
            set_phase_or_mode(mode_use)
            series = compute_agg_p(df_slice, mode=mode_use)
            return series.ffill().fillna(0.5)

    # If train/test subwindow specified, run calibration then test
    if args.train_start and args.train_end and args.test_start and args.test_end:
        train_mask = (df_full.index >= pd.to_datetime(args.train_start)) & (df_full.index <= pd.to_datetime(args.train_end))
        test_mask = (df_full.index >= pd.to_datetime(args.test_start)) & (df_full.index <= pd.to_datetime(args.test_end))
        df_tr = df_full.loc[train_mask]
        df_ts = df_full.loc[test_mask]
        if df_tr.empty or df_ts.empty:
            raise RuntimeError('Empty train/test slices')
        # Parse grids
        modes_grid = [m.strip() for m in (args.grid_modes or '').split(',') if m.strip()]
        k_grid = [float(x) for x in (args.grid_kappa or '').split(',') if x.strip()]
        def _parse_lin(s: str) -> list[float]:
            a, b, n = s.split(':');
            a = float(a); b = float(b); n = int(n)
            return [float(x) for x in np.linspace(a, b, max(2, n))]
        buys = _parse_lin(args.grid_buy)
        sells = _parse_lin(args.grid_sell)
        # Calibrate on train
        best = {'score': -1e18}
        for mode_use in modes_grid:
            for k in k_grid:
                agg_tr = _agg_series(df_tr, mode_use, k)
                for buy_thr in buys:
                    for sell_thr in sells:
                        strat_tr, m_tr = _evaluate_series(agg_tr, df_tr, buy_thr, sell_thr)
                        score = m_tr['Sharpe']
                        if score > best['score']:
                            best = {'score': score, 'mode': mode_use, 'kappa': float(k), 'buy': float(buy_thr), 'sell': float(sell_thr), 'train_metrics': m_tr}
        # Apply best on test
        agg_ts = _agg_series(df_ts, best['mode'], best['kappa'])
        strat_ts, m_ts = _evaluate_series(agg_ts, df_ts, best['buy'], best['sell'])
        # Build per-step log for test
        log_rows: List[Dict] = []
        pos = np.where(agg_ts.values >= best['buy'], 1.0, np.where(agg_ts.values <= best['sell'], 0.0, np.nan))
        pos = pd.Series(pos, index=agg_ts.index).ffill().fillna(0.0)
        pos = _apply_risk(df_ts, pos)
        ret = df_ts['Close'].pct_change().fillna(0.0)
        trade = pos.diff().abs().fillna(0.0)
        strat = pos.shift(1).fillna(0.0) * ret - args.fee * trade - args.slip * trade
        for i, d in enumerate(df_ts.index):
            nr = float(ret.iloc[i]) if i < len(ret) else np.nan
            sr = float(strat.iloc[i]) if i < len(strat) else np.nan
            dec = 'buy' if agg_ts.iloc[i] >= best['buy'] else ('sell' if agg_ts.iloc[i] <= best['sell'] else 'hold')
            acc = (1.0 if (np.isfinite(nr) and dec in ('buy','sell') and ((1.0 if dec=='buy' else -1.0) * nr) > 0) else (np.nan))
            log_rows.append(dict(date=pd.to_datetime(d).strftime('%Y-%m-%d %H:%M:%S'), close=float(df_ts['Close'].iloc[i]), agg_p=float(agg_ts.iloc[i]), decision=dec, position=float(pos.iloc[i]), next_ret=float(nr) if np.isfinite(nr) else np.nan, strat_ret=float(sr) if np.isfinite(sr) else np.nan, accuracy=acc, buy_thr=float(best['buy']), sell_thr=float(best['sell']), kappa=best['kappa'], mode=best['mode']))
        log_df = pd.DataFrame(log_rows)
        out_dir = args.out_dir or os.path.join('data', 'daily_eval', datetime.now().strftime('%Y%m%d_%H%M%S'))
        ensure_dir(out_dir)
        log_df.to_csv(os.path.join(out_dir, 'daily_log.csv'), index=False)
        summary = {
            'train': {'start': args.train_start, 'end': args.train_end, 'best': best},
            'test': {'start': args.test_start, 'end': args.test_end, 'metrics': m_ts},
            'risk': args.risk,
        }
        with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        print('Saved calibrated evaluation to', out_dir)
        print('Best (train):', {k: (round(v,3) if isinstance(v, float) else v) for k,v in best.items() if k!='train_metrics'})
        print('Test metrics:', {k: round(v,3) for k,v in m_ts.items()})
        return

    # Default: step-by-step log across [start, end]
    for i, d in enumerate(dates):
        # Use full history up to date d for feature context and transformer seq
        df_hist = df_full.loc[:d]
        if df_hist.shape[0] < max(2, args.seq):
            continue
        # Compute agg_p using only history up to d
        if args.use_transformer:
            agg_p = last_specs_agg_p(model, df_hist, args.seq, args.mode, kappa_prog, args.mc)
        else:
            agg_series = compute_agg_p(df_hist.tail(1), mode=args.mode)
            agg_p = float(agg_series.iloc[-1])

        # Decision to position
        if agg_p >= args.buy:
            pos = 1.0
            decision = 'buy'
        elif agg_p <= args.sell:
            pos = 0.0
            decision = 'sell'
        else:
            pos = pos_prev
            decision = 'hold'

        # Realized next-day return (if exists)
        # Next calendar trading day in full dataset
        full_dates = list(df_full.index)
        try:
            j_full = full_dates.index(d)
        except ValueError:
            j_full = None
        if j_full is not None and (j_full + 1) < len(full_dates):
            close_t = float(df_full.loc[full_dates[j_full], 'Close'])
            close_tp1 = float(df_full.loc[full_dates[j_full + 1], 'Close'])
            ret_next = (close_tp1 / max(close_t, 1e-12)) - 1.0
        else:
            ret_next = np.nan

        # Strategy return applied to next day
        turnover = abs(pos - pos_prev)
        strat_ret = (pos * ret_next) - args.fee * turnover - args.slip * turnover if np.isfinite(ret_next) else np.nan

        # Accuracy metric: sign match if there was a decisive signal (buy/sell)
        if np.isfinite(ret_next) and (decision in ('buy', 'sell')):
            pred_sign = 1.0 if decision == 'buy' else -1.0
            acc = 1.0 if (pred_sign * ret_next) > 0 else 0.0
        else:
            acc = np.nan

        row = dict(
            date=pd.to_datetime(d).strftime('%Y-%m-%d'),
            close=float(df_hist['Close'].iloc[-1]),
            agg_p=float(agg_p),
            decision=decision,
            position=float(pos),
            next_ret=float(ret_next) if np.isfinite(ret_next) else np.nan,
            strat_ret=float(strat_ret) if np.isfinite(strat_ret) else np.nan,
            accuracy=acc,
            buy_thr=float(args.buy),
            sell_thr=float(args.sell),
            kappa=kappa_prog,
            mode=args.mode,
        )
        log_rows.append(row)
        if np.isfinite(strat_ret) and (j_full is not None) and (j_full + 1) < len(full_dates):
            strat_returns.append((full_dates[j_full + 1], float(strat_ret)))
        pos_prev = pos

    log_df = pd.DataFrame(log_rows)
    # Build aligned strategy return series for metrics
    if strat_returns:
        idx, vals = zip(*strat_returns)
        strat_series = pd.Series(vals, index=pd.to_datetime(list(idx)))
    else:
        strat_series = pd.Series(dtype=float)

    summary = {
        'window': [args.start, args.end],
        'n_days': int(log_df.shape[0]),
        'n_trading_days': int(len(dates)),
        'hit_rate_on_decisive': float(np.nanmean(log_df['accuracy'].values)) if 'accuracy' in log_df and log_df['accuracy'].notna().any() else None,
        'strategy_metrics': metrics(strat_series),
    }

    out_dir = args.out_dir or os.path.join('data', 'daily_eval', datetime.now().strftime('%Y%m%d_%H%M%S'))
    ensure_dir(out_dir)
    log_df.to_csv(os.path.join(out_dir, 'daily_log.csv'), index=False)
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print('Saved daily evaluation to', out_dir)
    print('Hit-rate (decisive):', summary['hit_rate_on_decisive'])
    print('Strategy:', {k: round(v, 3) for k, v in summary['strategy_metrics'].items()})


if __name__ == '__main__':
    main()


