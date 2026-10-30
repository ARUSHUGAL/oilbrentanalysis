"""Brent Crude oil price analysis pipeline.

Downloads ten years of daily Brent Crude prices, flags statistical outliers,
and renders an annotated two-panel dashboard.

Usage:
    python pipeline.py
    python pipeline.py --start 2014-01-01 --end 2024-01-01 --output chart.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe: no display required, works in CI
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

TICKER = "BZ=F"
DEFAULT_START = "2014-01-01"
DEFAULT_END = "2024-01-01"
DEFAULT_OUTPUT = "oil_dashboard.png"

ANOMALY_SIGMA = 3.0
TREND_WINDOW = 90
VOLATILITY_WINDOW = 30

EVENTS = {
    "2014-11-28": "OPEC No-Cut Decision",
    "2016-02-11": "Price Hits 12-Year Low",
    "2020-03-09": "COVID + Saudi-Russia Price War",
    "2022-03-08": "Ukraine War Spike",
}

BG = "#0f0f0f"
GOLD = "#c8a951"
GRID = "#222222"
MUTED = "#aaaaaa"


def frame_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Reduce a yfinance download to a single 'Price' column indexed by date."""
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw.xs("Close", axis=1, level=0)
        series = close.iloc[:, 0]
    else:
        series = raw["Close"]

    df = series.rename("Price").to_frame()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


def download_prices(
    ticker: str = TICKER, start: str = DEFAULT_START, end: str = DEFAULT_END
) -> pd.DataFrame:
    """Fetch daily prices. Imports yfinance lazily so tests need no network."""
    import yfinance as yf

    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        raise RuntimeError(f"no data returned for {ticker} between {start} and {end}")
    return frame_from_raw(raw)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop missing rows and duplicate dates. Returns (frame, rows_dropped)."""
    before = len(df)
    out = df.dropna().sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out, before - len(out)


def detect_anomalies(df: pd.DataFrame, sigma: float = ANOMALY_SIGMA) -> pd.DataFrame:
    """Flag prices more than `sigma` standard deviations from the mean."""
    out = df.copy()
    std = out["Price"].std()
    if pd.isna(std) or std == 0:
        out["anomaly"] = False
        return out
    out["anomaly"] = (out["Price"] - out["Price"].mean()).abs() > sigma * std
    return out


def add_rolling_stats(
    df: pd.DataFrame,
    trend_window: int = TREND_WINDOW,
    vol_window: int = VOLATILITY_WINDOW,
) -> pd.DataFrame:
    """Attach a moving average and a rolling volatility column."""
    out = df.copy()
    out["rolling_avg"] = out["Price"].rolling(trend_window).mean()
    out["volatility"] = out["Price"].rolling(vol_window).std()
    return out


def nearest_price(df: pd.DataFrame, timestamp: pd.Timestamp) -> float:
    """Price on `timestamp`, or on the nearest trading day if it was a holiday."""
    if timestamp in df.index:
        return float(df.loc[timestamp, "Price"])
    position = df.index.get_indexer([timestamp], method="nearest")[0]
    return float(df.loc[df.index[position], "Price"])


def render_dashboard(
    df: pd.DataFrame,
    output: str | Path = DEFAULT_OUTPUT,
    events: dict[str, str] | None = None,
    title: str | None = None,
) -> Path:
    """Render the price + volatility dashboard to `output`."""
    events = EVENTS if events is None else events
    if title is None:
        title = f"Brent Crude Oil Price  |  {df.index.min().year}\u2013{df.index.max().year}"

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 9), gridspec_kw={"height_ratios": [3, 1]}, facecolor=BG
    )
    fig.subplots_adjust(hspace=0.08)

    # --- top panel: price, trend, anomalies, event markers ---
    ax1.set_facecolor(BG)
    ax1.plot(df.index, df["Price"], color=GOLD, linewidth=1.2,
             label="Brent Crude (USD/bbl)", zorder=3)
    ax1.fill_between(df.index, df["Price"], alpha=0.12, color=GOLD)
    ax1.plot(df.index, df["rolling_avg"], color="#ffffff", linewidth=1,
             linestyle="--", alpha=0.6, label=f"{TREND_WINDOW}-day Moving Avg")
    ax1.scatter(
        df.index[df["anomaly"]], df.loc[df["anomaly"], "Price"],
        color="red", zorder=5, s=30, label="Anomaly",
    )

    for date_str, label in events.items():
        timestamp = pd.to_datetime(date_str)
        price = nearest_price(df, timestamp)
        ax1.axvline(timestamp, color="#ff4444", linestyle="--", alpha=0.4, linewidth=0.8)
        ax1.text(timestamp, price + 3, label, fontsize=7, rotation=40,
                 color="#ff8888", ha="left", va="bottom")

    ax1.set_title(title, fontsize=15, color="white", fontfamily="monospace", pad=14)
    ax1.set_ylabel("USD per Barrel", color=MUTED, fontsize=10)
    ax1.tick_params(colors=MUTED, labelbottom=False)
    for spine in ax1.spines.values():
        spine.set_color("#333333")
    ax1.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax1.legend(facecolor="#1a1a1a", edgecolor="#333333", labelcolor="white", fontsize=8)

    # --- bottom panel: rolling volatility ---
    ax2.set_facecolor(BG)
    ax2.fill_between(df.index, df["volatility"], color="#5588ff", alpha=0.5,
                     label=f"{VOLATILITY_WINDOW}-day Volatility")
    ax2.plot(df.index, df["volatility"], color="#5588ff", linewidth=0.8)
    ax2.set_ylabel("Volatility (\u03c3)", color=MUTED, fontsize=10)
    ax2.set_xlabel("Year", color=MUTED, fontsize=10)
    ax2.tick_params(colors=MUTED)
    for spine in ax2.spines.values():
        spine.set_color("#333333")
    ax2.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.legend(facecolor="#1a1a1a", edgecolor="#333333", labelcolor="white", fontsize=8)

    output = Path(output)
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return output


def run(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    output: str | Path = DEFAULT_OUTPUT,
    ticker: str = TICKER,
) -> dict:
    """Execute the whole pipeline and return a summary of what happened."""
    raw = download_prices(ticker=ticker, start=start, end=end)
    df, dropped = clean(raw)
    if df.empty:
        raise RuntimeError("no usable rows left after cleaning")

    df = detect_anomalies(df)
    df = add_rolling_stats(df)
    anomalies = int(df["anomaly"].sum())
    path = render_dashboard(df, output=output)

    return {
        "rows": len(df),
        "dropped": dropped,
        "anomalies": anomalies,
        "start": str(df.index.min().date()),
        "end": str(df.index.max().date()),
        "output": str(path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default=DEFAULT_START, help="first date, YYYY-MM-DD")
    parser.add_argument("--end", default=DEFAULT_END, help="last date, YYYY-MM-DD")
    parser.add_argument("--ticker", default=TICKER, help="yfinance ticker (default BZ=F)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="path for the PNG")
    args = parser.parse_args(argv)

    summary = run(start=args.start, end=args.end, output=args.output, ticker=args.ticker)
    print(f"rows      : {summary['rows']} ({summary['dropped']} dropped)")
    print(f"range     : {summary['start']} -> {summary['end']}")
    print(f"anomalies : {summary['anomalies']}")
    print(f"saved     : {summary['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
