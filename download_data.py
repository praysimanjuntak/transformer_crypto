#!/usr/bin/env python3
"""
Download historical kline data from Binance.

Usage:
    python download_data.py --pair BTCUSDT --timeframe 15m --start 2020-01 --end 2025-11
"""

import argparse
import os
import subprocess
import zipfile
from pathlib import Path


HEADER = "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore"
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"


def parse_args():
    parser = argparse.ArgumentParser(description="Download Binance historical data")
    parser.add_argument("--pair", type=str, default="BTCUSDT", help="Trading pair (e.g., BTCUSDT)")
    parser.add_argument("--timeframe", type=str, default="15m", help="Timeframe (1m, 5m, 15m, 1h, 4h, 1d)")
    parser.add_argument("--start", type=str, default="2020-01", help="Start month (YYYY-MM)")
    parser.add_argument("--end", type=str, default="2025-11", help="End month (YYYY-MM)")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    return parser.parse_args()


def generate_months(start: str, end: str):
    """Generate list of months between start and end."""
    start_year, start_month = map(int, start.split("-"))
    end_year, end_month = map(int, end.split("-"))

    months = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        months.append(f"{year}-{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


def download_and_process(pair: str, timeframe: str, month: str, output_dir: Path):
    """Download, extract, and add headers to a single month's data."""
    filename = f"{pair}-{timeframe}-{month}"
    zip_file = f"{filename}.zip"
    csv_file = f"{filename}.csv"
    url = f"{BASE_URL}/{pair}/{timeframe}/{zip_file}"

    csv_path = output_dir / csv_file

    # Skip if already exists
    if csv_path.exists():
        print(f"  {month}: already exists, skipping")
        return True

    # Download
    print(f"  {month}: downloading...", end=" ", flush=True)
    result = subprocess.run(
        ["curl", "-sf", "-O", url],
        capture_output=True,
        cwd=output_dir
    )

    if result.returncode != 0:
        print("not available, skipping")
        return False

    # Unzip
    zip_path = output_dir / zip_file
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(output_dir)
        print("extracted...", end=" ", flush=True)
    except Exception as e:
        print(f"unzip failed: {e}")
        zip_path.unlink(missing_ok=True)
        return False

    # Check if header exists and add if needed
    with open(csv_path, 'r') as f:
        first_line = f.readline().strip()

    if not first_line.startswith("open_time"):
        with open(csv_path, 'r') as f:
            content = f.read()
        with open(csv_path, 'w') as f:
            f.write(HEADER + "\n" + content)
        print("added header...", end=" ", flush=True)

    # Clean up zip
    zip_path.unlink()
    print("done")
    return True


def main():
    args = parse_args()

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(args.pair.lower()) / args.timeframe

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate months
    months = generate_months(args.start, args.end)
    print(f"Downloading {len(months)} months of {args.pair} {args.timeframe} data...")
    print(f"Output: {output_dir}")

    # Download each month
    success = 0
    for month in months:
        if download_and_process(args.pair, args.timeframe, month, output_dir):
            success += 1

    # Summary
    print(f"\nDownloaded {success}/{len(months)} files to {output_dir}")

    # Count total rows
    csv_files = sorted(output_dir.glob("*.csv"))
    total_rows = 0
    for f in csv_files:
        with open(f) as file:
            total_rows += sum(1 for _ in file) - 1  # minus header

    print(f"Total candles: {total_rows:,}")


if __name__ == "__main__":
    main()
