from __future__ import annotations

import os
import time
import json
import math
import datetime as dt
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ---------- STOOQ (daily) ----------

def _stooq_symbol(ticker: str) -> Optional[str]:
    t = ticker.strip().upper()
    if t.endswith('-USD'):
        # Stooq supports some crypto pairs like btcusd
        base = t.replace('-USD', '').lower()
        return f"{base}usd"
    if t.startswith('^'):
        # indices often have direct names; try without ^ and .us
        return t[1:].lower()
    # US ETFs/stocks
    return f"{t.lower()}.us"


def fetch_stooq_daily(ticker: str) -> Optional[pd.DataFrame]:
    sym = _stooq_symbol(ticker)
    if not sym:
        return None
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200 or 'Date,Open,High,Low,Close,Volume' not in r.text:
            return None
        df = pd.read_csv(pd.compat.StringIO(r.text))
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        return df
    except Exception:
        return None


# ---------- BINANCE (crypto intraday/daily) ----------

BINANCE_MAP: Dict[str, str] = {
    'BTC-USD': 'BTCUSDT', 'ETH-USD': 'ETHUSDT', 'BNB-USD': 'BNBUSDT', 'SOL-USD': 'SOLUSDT',
    'ADA-USD': 'ADAUSDT', 'XRP-USD': 'XRPUSDT', 'LTC-USD': 'LTCUSDT', 'DOGE-USD': 'DOGEUSDT',
    'AVAX-USD': 'AVAXUSDT', 'MATIC-USD': 'MATICUSDT'
}


def _to_ms(d: dt.datetime) -> int:
    return int(d.timestamp() * 1000)


def fetch_binance_klines(ticker: str, interval: str, start: str, end: str) -> Optional[pd.DataFrame]:
    sym = BINANCE_MAP.get(ticker.upper())
    if not sym:
        return None
    base = "https://api.binance.com/api/v3/klines"
    start_ms = _to_ms(dt.datetime.fromisoformat(start))
    end_ms = _to_ms(dt.datetime.fromisoformat(end))
    rows: List[List] = []
    limit = 1000
    cur = start_ms
    while cur < end_ms:
        params = {"symbol": sym, "interval": interval, "startTime": cur, "endTime": end_ms, "limit": limit}
        try:
            r = requests.get(base, params=params, timeout=15)
            if r.status_code != 200:
                break
            chunk = r.json()
            if not chunk:
                break
            rows.extend(chunk)
            next_start = int(chunk[-1][0]) + 1
            if next_start <= cur:
                break
            cur = next_start
            time.sleep(0.3)
        except Exception:
            break
    if not rows:
        return None
    cols = ['OpenTime','Open','High','Low','Close','Volume','CloseTime','QAV','Trades','TBB','TBQ','Ignore']
    df = pd.DataFrame(rows, columns=cols)
    df['Date'] = pd.to_datetime(df['OpenTime'], unit='ms')
    out = df[['Date','Open','High','Low','Close','Volume']].copy()
    out[['Open','High','Low','Close','Volume']] = out[['Open','High','Low','Close','Volume']].astype(float)
    out = out.set_index('Date').sort_index()
    return out


def save_to_csv(ticker: str, df: pd.DataFrame, out_dir: str) -> str:
    _ensure_dir(out_dir)
    fp = os.path.join(out_dir, f"{ticker}.csv")
    df.reset_index().rename(columns={'index': 'Date'}).to_csv(fp, index=False)
    return fp


# ---------- TIINGO ----------

def fetch_tiingo_data(ticker: str, start: str, end: str, api_token: str, interval: str = 'daily') -> Optional[pd.DataFrame]:
    """
    Fetch data from Tiingo API.
    interval: 'daily', 'hourly', '1min', '5min', '15min', '30min'
    """
    if not api_token:
        return None
    
    # Clean ticker
    t = ticker.upper()
    if t.endswith('-USD'):
        t = t.replace('-USD', '')
    if t.startswith('^'):
        t = t[1:]
    
    if interval == 'daily':
        url = f"https://api.tiingo.com/tiingo/daily/{t}/prices"
        params = {
            'startDate': start,
            'endDate': end,
            'token': api_token
        }
    else:
        # Intraday
        freq_map = {
            'hourly': '1hour',
            '1min': '1min',
            '5min': '5min', 
            '15min': '15min',
            '30min': '30min'
        }
        freq = freq_map.get(interval, '1hour')
        url = f"https://api.tiingo.com/iex/{t}/prices"
        params = {
            'startDate': start,
            'endDate': end,
            'resampleFreq': freq,
            'token': api_token
        }
    
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            return None
        
        data = r.json()
        if not data:
            return None
            
        df = pd.DataFrame(data)
        if 'date' in df.columns:
            df['Date'] = pd.to_datetime(df['date'])
        elif 'datetime' in df.columns:
            df['Date'] = pd.to_datetime(df['datetime'])
        else:
            return None
            
        # Map columns
        col_map = {}
        if 'open' in df.columns:
            col_map['open'] = 'Open'
        if 'high' in df.columns:
            col_map['high'] = 'High'
        if 'low' in df.columns:
            col_map['low'] = 'Low'
        if 'close' in df.columns:
            col_map['close'] = 'Close'
        if 'volume' in df.columns:
            col_map['volume'] = 'Volume'
            
        df = df.rename(columns=col_map)
        
        # Keep relevant columns
        keep_cols = ['Date']
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                keep_cols.append(col)
                
        df = df[keep_cols].set_index('Date').sort_index()
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df
        
    except Exception as e:
        print(f"Tiingo error for {ticker}: {e}")
        return None


def collect_to_csv(tickers: List[str], start: str, end: str, out_dir: str, daily: bool = True, intraday_interval: Optional[str] = None, tiingo_token: Optional[str] = None) -> Dict[str, str]:
    written: Dict[str, str] = {}
    for t in tickers:
        df: Optional[pd.DataFrame] = None
        
        # Try Tiingo first if token provided
        if tiingo_token:
            interval = 'daily' if daily else (intraday_interval or 'hourly')
            df = fetch_tiingo_data(t, start, end, tiingo_token, interval)
        
        # Try Stooq daily if no Tiingo data
        if df is None and daily:
            df = fetch_stooq_daily(t)
        
        # Try Binance for crypto if still no data
        if df is None and intraday_interval is not None:
            df = fetch_binance_klines(t, intraday_interval, start, end)
            
        if df is None:
            continue
            
        # Trim to range - handle timezone aware/naive comparison
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        if df.index.tz is not None:
            # DataFrame index is timezone-aware, make comparison values timezone-aware too
            if start_dt.tz is None:
                start_dt = start_dt.tz_localize('UTC')
            if end_dt.tz is None:
                end_dt = end_dt.tz_localize('UTC')
        else:
            # DataFrame index is timezone-naive, make comparison values timezone-naive
            if start_dt.tz is not None:
                start_dt = start_dt.tz_localize(None)
            if end_dt.tz is not None:
                end_dt = end_dt.tz_localize(None)
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        if df.empty:
            continue
            
        fp = save_to_csv(t, df[['Close']], out_dir)
        written[t] = fp
        
    return written


