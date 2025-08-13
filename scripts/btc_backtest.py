#!/usr/bin/env python3
import numpy as np
import pandas as pd
import yfinance as yf
from probstates.markets import (
    sma, momentum, rsi, indicator_to_prob, FeatureSpec, aggregate_specs,
    atr, macd, bollinger, realized_vol, fetch_yf_news_sentiment, sentiment_to_phase,
    fetch_prices, cross_asset_features
)
from probstates.market_cache import fetch_prices_cached, fetch_news_sentiment_cached
from probstates import set_phase_or_mode
import argparse


def _synthetic_btc(start='2018-01-01', end='2025-01-01', n: int | None = None) -> pd.DataFrame:
    rng = pd.date_range(start=start, end=end, freq='D') if n is None else pd.date_range(end=end, periods=n, freq='D')
    dt = 1.0/252.0
    mu, sigma = 0.2, 0.6
    close = [20000.0]
    rs = np.random.RandomState(42)
    for _ in range(len(rng)-1):
        z = rs.randn()
        close.append(close[-1] * float(np.exp((mu - 0.5*sigma*sigma)*dt + sigma*np.sqrt(dt)*z)))
    close = np.array(close)
    vol = np.clip(np.abs(rs.randn(len(rng))) * 0.01, 0.002, 0.05)
    high = close * (1.0 + vol)
    low = close * (1.0 - vol)
    return pd.DataFrame({'Close': close, 'High': high, 'Low': low}, index=rng)


def load_btc(start='2018-01-01', end='2025-01-01'):
    df = yf.download('BTC-USD', start=start, end=end, progress=False)
    df = df[['Close']].dropna().copy()
    df['mom10'] = momentum(df['Close'].values, 10)
    df['rsi14'] = rsi(df['Close'].values, 14)
    df['sma20'] = sma(df['Close'].values, 20)
    df['sma50'] = sma(df['Close'].values, 50)
    # Drop NaNs only for core indicators to avoid wiping out rows due to sparse ctx features
    core_cols = ['Close','mom10','rsi14','sma20','sma50','bb_pb','macd_hist','atr14','rv20']
    df.dropna(subset=[c for c in core_cols if c in df.columns], inplace=True)
    return df


def build_specs(row):
    """Row is a pandas Series."""
    def get(name: str) -> float:
        return float(row[name])

    specs = []
    mom = get('mom10')
    p_mom = float(indicator_to_prob(np.array([mom]))[-1])
    phi_mom = 0.0 if mom >= 0.0 else np.pi
    specs.append(FeatureSpec('momentum_10', p_mom, phi_mom, 2))
    rsi_v = get('rsi14')
    p_rsi = float(indicator_to_prob(np.array([rsi_v]), center=50.0, k=1.2)[-1])
    phi_rsi = 0.0 if rsi_v >= 50.0 else np.pi
    specs.append(FeatureSpec('rsi_14', p_rsi, phi_rsi, 1))
    cross = get('sma20') - get('sma50')
    p_cross = float(indicator_to_prob(np.array([cross]))[-1])
    phi_cross = 0.0 if cross >= 0 else np.pi
    specs.append(FeatureSpec('sma20_vs_50', p_cross, phi_cross, 2))
    # Bollinger %B: high -> bullish
    pb = get('bb_pb')
    p_pb = float(indicator_to_prob(np.array([pb]), center=0.5, k=1.2)[-1])
    phi_pb = 0.0 if pb >= 0.5 else np.pi
    specs.append(FeatureSpec('bb_pb', p_pb, phi_pb, 1))
    # MACD hist
    mh = get('macd_hist')
    p_mh = float(indicator_to_prob(np.array([mh]))[-1])
    phi_mh = 0.0 if mh >= 0 else np.pi
    specs.append(FeatureSpec('macd_hist', p_mh, phi_mh, 1))
    # Volatility filter: high rv -> lower confidence
    rv = get('rv20')
    conf = float(np.clip(1.0 - float(np.tanh(abs(rv))), 0.0, 1.0))
    specs.append(FeatureSpec('rv_conf', conf, 0.0 if conf >= 0.5 else np.pi, 1))
    # News sentiment
    if 'news_sent' in row:
        s = float(row['news_sent'])
        p_news = 0.5 + 0.5 * abs(s)
        phi_news = sentiment_to_phase(s)
        specs.append(FeatureSpec('news', p_news, phi_news, 2))
    # Cross-asset features: combine corr & momentum
    for key, val in row.items():
        if isinstance(key, str) and key.startswith('corr20_'):
            t = key.split('corr20_')[-1]
            mom_key = f'mom20_{t}'
            if mom_key in row:
                corr = float(val)
                momv = float(row[mom_key])
                if np.isnan(corr) or np.isnan(momv):
                    continue
                score = corr * momv
                p = float(indicator_to_prob(np.array([score]), k=1.0)[-1])
                phi = 0.0 if score >= 0.0 else np.pi
                specs.append(FeatureSpec(f'ctx_{t}', p, phi, 1))
        if isinstance(key, str) and key.startswith('risk_'):
            r = float(val)
            if np.isnan(r):
                continue
            p_r = float(indicator_to_prob(np.array([r]), k=1.0)[-1])
            phi_r = 0.0 if r >= 0.0 else np.pi
            specs.append(FeatureSpec(key, p_r, phi_r, 1))
    return specs


def compute_agg_p(df: pd.DataFrame, mode: str = 'weight') -> pd.Series:
    set_phase_or_mode(mode)
    return pd.Series(
        (aggregate_specs(build_specs(r), mode=mode).probability for _, r in df.iterrows()),
        index=df.index,
    )


def backtest(df: pd.DataFrame, buy_thr=0.55, sell_thr=0.45, fee=0.001, mode: str = 'weight'):
    agg_p = compute_agg_p(df, mode=mode)
    df = df.copy()
    df['agg_p'] = agg_p.values
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
    trade = df['pos'].diff().abs().fillna(0.0)
    strat = df['pos'].shift(1).fillna(0.0) * ret - fee * trade
    return df, strat


def metrics(series: pd.Series):
    x = series.dropna().values
    if x.size == 0:
        return dict(CAGR=0.0, Sharpe=0.0, MDD=0.0)
    ann = np.prod(1.0 + x) ** (252/len(x)) - 1.0
    vol = np.std(x) * np.sqrt(252)
    sharpe = ann / (vol + 1e-12)
    curve = (1.0 + x).cumprod()
    peak = np.maximum.accumulate(curve)
    mdd = float(np.max((peak - curve)/peak))
    return dict(CAGR=ann, Sharpe=sharpe, MDD=mdd)


def optimize_thresholds(df_train: pd.DataFrame, fee: float, mode: str) -> tuple[float, float, float]:
    """Grid-search thresholds maximizing Sharpe on train segment."""
    agg_p = compute_agg_p(df_train, mode=mode)
    ret = df_train['Close'].pct_change().fillna(0.0)
    best = (-1e9, 0.55, 0.45)
    for buy in np.linspace(0.52, 0.65, 14):
        for sell in np.linspace(0.35, 0.48, 14):
            pos = np.where(agg_p.values >= buy, 1.0, np.where(agg_p.values <= sell, 0.0, np.nan))
            # forward-fill holds, start from 0
            pos = pd.Series(pos, index=agg_p.index).fillna(method='ffill').fillna(0.0)
            trade = pos.diff().abs().fillna(0.0)
            strat = pos.shift(1).fillna(0.0) * ret - fee * trade
            m = metrics(strat)
            score = m['Sharpe']
            if score > best[0]:
                best = (score, buy, sell)
    return best[1], best[2], best[0]


def _build_feature_matrix_like_train(df: pd.DataFrame) -> np.ndarray:
    base_cols = ['mom10','rsi14','sma20','sma50','bb_pb','macd_hist','atr14','rv20']
    cols = [c for c in base_cols if c in df.columns]
    ctx_cols = sorted([c for c in df.columns if isinstance(c, str) and (c.startswith('corr20_') or c.startswith('mom20_') or c.startswith('risk_'))])
    cols.extend(ctx_cols)
    return df[cols].values.astype(float)


def backtest_with_transformer(df: pd.DataFrame, model_path: str, seq_len: int = 64, fee: float = 0.001, mode: str = 'weight'):
    try:
        from probstates.transformer import load_model
        from probstates.markets import aggregate_specs
    except Exception:
        print('Transformer dependencies are not available.')
        return df.copy(), pd.Series(dtype=float)
    try:
        model = load_model(model_path)
    except Exception:
        print('Cannot load transformer model at', model_path)
        return df.copy(), pd.Series(dtype=float)
    set_phase_or_mode(mode)
    X = _build_feature_matrix_like_train(df)
    probs = []
    for i in range(len(df)):
        if i + 1 < seq_len:
            probs.append(np.nan)
            continue
        last_features = X[i]
        specs = model.build_specs(last_features)
        agg = aggregate_specs(specs, mode=mode)
        probs.append(agg.probability)
    out = df.copy()
    out['agg_p'] = pd.Series(probs, index=out.index).ffill().fillna(0.5)
    ret = out['Close'].pct_change().fillna(0.0)
    pos = (out['agg_p'] >= 0.55).astype(float)
    pos = pd.Series(pos, index=out.index).ffill().fillna(0.0)
    trade = pos.diff().abs().fillna(0.0)
    strat = pos.shift(1).fillna(0.0) * ret - fee * trade
    return out, strat


def walk_forward(df: pd.DataFrame, fee: float = 0.001, mode: str = 'weight'):
    years = sorted(set(df.index.year))
    results = []
    all_strat = []
    for i in range(1, len(years)):
        train_year = years[i-1]
        test_year = years[i]
        train = df[df.index.year == train_year]
        test = df[df.index.year == test_year]
        if len(train) < 100 or len(test) < 50:
            continue
        buy_thr, sell_thr, sharpe_tr = optimize_thresholds(train, fee=fee, mode=mode)
        _, strat = backtest(test, buy_thr=buy_thr, sell_thr=sell_thr, fee=fee, mode=mode)
        res = dict(year=test_year, buy=buy_thr, sell=sell_thr)
        res.update({f'test_{k}': v for k, v in metrics(strat).items()})
        res['train_sharpe'] = sharpe_tr
        results.append(res)
        all_strat.append(strat)
    if all_strat:
        joined = pd.concat(all_strat).sort_index()
    else:
        joined = pd.Series(dtype=float)
    return pd.DataFrame(results), joined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2018-01-01')
    ap.add_argument('--end', default='2025-01-01')
    ap.add_argument('--fee', type=float, default=0.001)
    ap.add_argument('--mode', default='weight', choices=['weight','norm','quant','opt'])
    ap.add_argument('--wf', action='store_true', help='run walk-forward optimization')
    ap.add_argument('--ctx', default='DX-Y.NYB,^VIX,SPY,GLD,ETH-USD', help='comma-separated cross-asset tickers')
    ap.add_argument('--synthetic', action='store_true', help='use synthetic BTC if network/cache empty')
    ap.add_argument('--use_transformer', action='store_true', help='use trained transformer for adaptive agg')
    ap.add_argument('--model_path', default='data/models/transformer.pt')
    args = ap.parse_args()

    # Robust BTC data load (fallback to cache if yfinance returns empty)
    raw = yf.download('BTC-USD', start=args.start, end=args.end, progress=False, auto_adjust=False)
    df = None
    if isinstance(raw, pd.DataFrame) and not raw.empty:
        # Handle possible MultiIndex and missing 'Close'
        if isinstance(raw.columns, pd.MultiIndex):
            close_col = None
            for cand in [("Close", "BTC-USD"), ("Adj Close", "BTC-USD")]:
                if cand in raw.columns:
                    close_col = cand
                    break
            if close_col is not None:
                out = pd.DataFrame({'Close': raw[close_col]})
                # try High/Low too
                for name in ['High','Low']:
                    col_cand = (name, 'BTC-USD')
                    if col_cand in raw.columns:
                        out[name] = raw[col_cand]
                df = out.dropna(subset=['Close'])
        else:
            cols = list(raw.columns)
            if 'Close' in cols or 'Adj Close' in cols:
                close_name = 'Close' if 'Close' in cols else 'Adj Close'
                out = pd.DataFrame({'Close': raw[close_name]})
                if 'High' in cols and 'Low' in cols:
                    out['High'] = raw['High']
                    out['Low'] = raw['Low']
                df = out.dropna(subset=['Close'])
    if df is None or df.empty:
        cached = fetch_prices_cached(['BTC-USD'], args.start, args.end, column='Close')
        s = cached.get('BTC-USD')
        if s is not None and len(s) > 0:
            df = pd.DataFrame({'Close': s}).dropna()
        else:
            # synthetic fallback
            if args.synthetic:
                df = _synthetic_btc(args.start, args.end)
                print('Using synthetic BTC series (no network/cache).')
            else:
                print('No BTC data available (network/cache). Try --synthetic to run offline.')
                print(pd.DataFrame())
                return
    df['mom10'] = momentum(df['Close'].values, 10)
    df['rsi14'] = rsi(df['Close'].values, 14)
    df['sma20'] = sma(df['Close'].values, 20)
    df['sma50'] = sma(df['Close'].values, 50)
    mid, up, lo, pb = bollinger(df['Close'].values, 20)
    df['bb_pb'] = pb
    macd_line, signal_line, hist = macd(df['Close'].values)
    df['macd'] = macd_line; df['macd_hist'] = hist
    if 'High' in df.columns and 'Low' in df.columns:
        df['atr14'] = atr(df['High'].values, df['Low'].values, df['Close'].values, 14)
    else:
        df['atr14'] = np.nan
    df['rv20'] = realized_vol(df['Close'].values, 20)
    # Attach daily news sentiment (coarse)
    sent = fetch_yf_news_sentiment('BTC-USD', lookback_days=365*3)
    if sent:
        s_series = pd.Series(sent)
        s_series.index = pd.to_datetime(s_series.index)
        daily_s = s_series.resample('D').mean().ffill()
        df['news_sent'] = daily_s.reindex(df.index, method='ffill').fillna(0.0)
    else:
        df['news_sent'] = 0.0
    # Attach daily news sentiment via yfinance headlines (coarse)
    s_series = fetch_news_sentiment_cached('BTC-USD', 365*3, fetch_yf_news_sentiment)
    if not s_series.empty:
        s_series.index = pd.to_datetime(s_series.index)
        daily_s = s_series.resample('D').mean().ffill()
        df['news_sent'] = daily_s.reindex(df.index, method='ffill').fillna(0.0)
    else:
        df['news_sent'] = 0.0
    # Cross-asset context features
    tickers = [t.strip() for t in (args.ctx or '').split(',') if t.strip()]
    if tickers:
        ctx_raw = fetch_prices_cached(tickers, args.start, args.end, column='Close')
        ctx = {k: v.values if isinstance(v, pd.Series) else v for k, v in ctx_raw.items()}
        if ctx:
            feats = cross_asset_features(df['Close'].values, ctx, windows=(20,60))
            for k, arr in feats.items():
                df[k] = arr
    # Only require core indicators; allow NaNs in context features to avoid wiping out rows
    core_cols = ['Close','mom10','rsi14','sma20','sma50','bb_pb','macd_hist','rv20']
    if ('High' in df.columns) and ('Low' in df.columns) and ('atr14' in df.columns):
        core_cols.append('atr14')
    df.dropna(subset=[c for c in core_cols if c in df.columns], inplace=True)
    if df.empty:
        print('No usable rows after feature computation.')
        print(pd.DataFrame())
        return

    if args.wf:
        table, joined = walk_forward(df, fee=args.fee, mode=args.mode)
        print(table)
        if not joined.empty:
            print('WF total:', {k: round(v,3) for k, v in metrics(joined).items()})
    else:
        if args.use_transformer:
            df2, strat = backtest_with_transformer(df, args.model_path, seq_len=64, fee=args.fee, mode=args.mode)
            bench = df2['Close'].pct_change().fillna(0.0)
            print('Strategy (Transformer):', {k: round(v,3) for k,v in metrics(strat).items()})
            print('Buy&Hold:', {k: round(v,3) for k,v in metrics(bench).items()})
        else:
            df2, strat = backtest(df, mode=args.mode, fee=args.fee)
            bench = df2['Close'].pct_change().fillna(0.0)
            print('Strategy:', {k: round(v,3) for k,v in metrics(strat).items()})
            print('Buy&Hold:', {k: round(v,3) for k,v in metrics(bench).items()})


if __name__ == '__main__':
    main()


