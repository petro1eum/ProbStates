#!/usr/bin/env python3
import argparse
import os
import pandas as pd
from typing import List

from probstates.data_sources import collect_to_csv


def merge_incremental(fp: str, df_new: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    if os.path.exists(fp):
        try:
            df_old = pd.read_csv(fp, parse_dates=['Date'])
            both = pd.concat([df_old, df_new.reset_index().rename(columns={'index': 'Date'})], ignore_index=True)
            both = both.drop_duplicates(subset=['Date']).sort_values('Date')
            both.to_csv(fp, index=False)
            return
        except Exception:
            pass
    df_new.reset_index().rename(columns={'index': 'Date'}).to_csv(fp, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tickers', required=True, help='comma-separated list of tickers')
    ap.add_argument('--start', default='2005-01-01')
    ap.add_argument('--end', default='2025-01-01')
    ap.add_argument('--interval', default='1d', choices=['1d','1h','30m','15m'])
    ap.add_argument('--out_dir', default='data/cache_tiingo')
    ap.add_argument('--tiingo_token', default=None)
    args = ap.parse_args()

    tickers: List[str] = [t.strip() for t in args.tickers.split(',') if t.strip()]
    daily = args.interval == '1d'
    intr = None if daily else args.interval
    written = collect_to_csv(
        tickers, args.start, args.end, args.out_dir,
        daily=daily, intraday_interval=intr, tiingo_token=args.tiingo_token
    )
    # Merge incrementally if files already exist (collect_to_csv writes new snapshots)
    for t, fp in written.items():
        try:
            df = pd.read_csv(fp, parse_dates=['Date']).set_index('Date')
            merge_incremental(fp, df)
        except Exception:
            continue
    print('Cached to', args.out_dir, 'tickers:', len(written))


if __name__ == '__main__':
    main()


