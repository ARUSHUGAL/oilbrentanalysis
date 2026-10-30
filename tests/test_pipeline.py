"""Tests for the Brent Crude pipeline. No network access required."""
import numpy as np
import pandas as pd
import pytest

import pipeline


def make_frame(values, start="2020-01-01"):
    """Build a Date-indexed 'Price' frame from a list of floats."""
    index = pd.bdate_range(start=start, periods=len(values))
    return pd.DataFrame({"Price": values}, index=index)


def make_raw(values, start="2020-01-01", multiindex=False):
    index = pd.bdate_range(start=start, periods=len(values))
    if multiindex:
        columns = pd.MultiIndex.from_product([["Close", "High"], ["BZ=F"]])
        data = np.column_stack([values, values])
        return pd.DataFrame(data, index=index, columns=columns)
    return pd.DataFrame({"Close": values}, index=index)


class TestFrameFromRaw:
    def test_flat_columns_become_price(self):
        raw = make_raw([70.0, 71.0, 72.0])
        df = pipeline.frame_from_raw(raw)
        assert list(df.columns) == ["Price"]
        assert df.index.name == "Date"
        assert df["Price"].tolist() == [70.0, 71.0, 72.0]

    def test_multiindex_columns_are_flattened(self):
        raw = make_raw([70.0, 71.0, 72.0], multiindex=True)
        df = pipeline.frame_from_raw(raw)
        assert list(df.columns) == ["Price"]
        assert len(df) == 3


class TestClean:
    def test_drops_missing_and_counts_them(self):
        df = make_frame([70.0, np.nan, 72.0, np.nan, 74.0])
        cleaned, dropped = pipeline.clean(df)
        assert dropped == 2
        assert len(cleaned) == 3
        assert not cleaned["Price"].isna().any()

    def test_sorts_and_deduplicates_dates(self):
        df = make_frame([1.0, 2.0, 3.0])
        shuffled = pd.concat([df.iloc[[2]], df.iloc[[0]], df.iloc[[0]]])
        cleaned, dropped = pipeline.clean(shuffled)
        assert cleaned.index.is_monotonic_increasing
        assert len(cleaned) == 2
        assert dropped == 1


class TestDetectAnomalies:
    def test_flags_a_single_extreme_point(self):
        values = [50.0] * 20 + [200.0]
        df = pipeline.detect_anomalies(make_frame(values))
        assert df["anomaly"].sum() == 1
        assert bool(df["anomaly"].iloc[-1]) is True

    def test_flat_series_has_no_anomalies(self):
        df = pipeline.detect_anomalies(make_frame([60.0] * 10))
        assert df["anomaly"].sum() == 0

    def test_negative_sigma_is_stricter(self):
        values = [50.0, 51.0, 52.0, 60.0]
        loose = pipeline.detect_anomalies(make_frame(values), sigma=10)
        tight = pipeline.detect_anomalies(make_frame(values), sigma=0.5)
        assert tight["anomaly"].sum() >= loose["anomaly"].sum()


class TestRollingStats:
    def test_moving_average_matches_pandas(self):
        df = pipeline.add_rolling_stats(make_frame([1.0, 2.0, 3.0, 4.0, 5.0]),
                                       trend_window=3, vol_window=2)
        assert df["rolling_avg"].iloc[2] == pytest.approx(2.0)
        assert pd.isna(df["rolling_avg"].iloc[0])
        # rolling(2) at index 2 covers [2, 3], not [1, 2, 3]
        assert df["volatility"].iloc[2] == pytest.approx(pd.Series([2.0, 3.0]).std())

    def test_does_not_mutate_input(self):
        original = make_frame([1.0, 2.0, 3.0, 4.0])
        pipeline.add_rolling_stats(original, trend_window=2, vol_window=2)
        assert "rolling_avg" not in original.columns


class TestNearestPrice:
    def test_exact_date(self):
        df = make_frame([10.0, 11.0, 12.0], start="2020-01-06")
        assert pipeline.nearest_price(df, pd.Timestamp("2020-01-07")) == 11.0

    def test_falls_back_to_closest_trading_day(self):
        df = make_frame([10.0, 11.0, 12.0], start="2020-01-06")
        price = pipeline.nearest_price(df, pd.Timestamp("2020-01-07 20:00"))
        assert price in (11.0, 12.0)


class TestRender:
    def test_writes_a_png(self, tmp_path):
        df = pipeline.add_rolling_stats(
            pipeline.detect_anomalies(make_frame([60.0] * 40 + [90.0] + [60.0] * 40)),
            trend_window=5, vol_window=5,
        )
        out = pipeline.render_dashboard(df, output=tmp_path / "chart.png")
        assert out.exists()
        assert out.stat().st_size > 5000
        assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


class TestRun:
    def test_end_to_end_with_stubbed_download(self, monkeypatch, tmp_path):
        values = [60.0] * 50 + [150.0] + [60.0] * 49
        monkeypatch.setattr(pipeline, "download_prices",
                            lambda **_: pipeline.frame_from_raw(make_raw(values)))
        summary = pipeline.run(output=tmp_path / "out.png")

        assert summary["rows"] == 100
        assert summary["dropped"] == 0
        assert summary["anomalies"] >= 1
        assert (tmp_path / "out.png").exists()

    def test_empty_download_raises(self, monkeypatch):
        monkeypatch.setattr(pipeline, "download_prices", lambda **_: pd.DataFrame())
        with pytest.raises(RuntimeError):
            pipeline.run(output="unused.png")
