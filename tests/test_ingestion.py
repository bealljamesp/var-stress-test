"""Offline tests for stress_engine.ingestion using synthetic HTML fixtures."""

import pandas as pd
import pytest

from stress_engine.ingestion import (
    assess_coverage,
    classify_cadence,
    clean_series,
    extract_distressed_patches,
    parse_raw_records,
    process_entity,
    TARGET_ENTITIES,
)


def _record(date: str, price: str) -> str:
    epoch = int(pd.Timestamp(date).timestamp())
    return f'{{"d": {epoch}, "v": {price}}}'


def _html(records: list[tuple[str, str]]) -> str:
    body = ",".join(_record(d, p) for d, p in records)
    return f"<html><script>var data = [{body}];</script></html>"


def _monthly(start: str, end: str, base: float = 10.0) -> list[tuple[str, str]]:
    dates = pd.date_range(start, end, freq=pd.offsets.MonthEnd())
    return [(d.strftime("%Y-%m-%d"), f"{base + i * 0.1:.4f}") for i, d in enumerate(dates)]


def _entity(ticker: str):
    return next(e for e in TARGET_ENTITIES if e.ticker == ticker)


# =====================================================================
# Parsing
# =====================================================================


def test_parse_matches_quoted_and_unquoted_prices() -> None:
    content = (
        '{"d": 1199145600, "v": 10.5} '
        '{"d": 1201824000, "v": "11.25"} '
        '{ "d" : 1204329600 , "v" : 12 }'
    )
    df = parse_raw_records(content)
    assert len(df) == 3
    assert df["Close"].tolist() == [10.5, 11.25, 12.0]


def test_parse_rejects_malformed_price_instead_of_crashing() -> None:
    # A price with two decimal points is not a valid record and must not match
    content = '{"d": 1199145600, "v": 1.2.3}'
    df = parse_raw_records(content)
    assert df.empty


def test_parse_returns_empty_frame_when_no_records() -> None:
    df = parse_raw_records("<html>no data</html>")
    assert df.empty
    assert list(df.columns) == ["Date", "Close"]


# =====================================================================
# Cleaning
# =====================================================================


def test_clean_drops_non_positive_and_duplicate_dates() -> None:
    raw = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2008-01-31", "2008-01-31", "2008-02-29", "2008-03-31"]),
            "Close": [10.0, 10.5, 0.0, 11.0],
        }
    )
    cleaned, invalid, dups = clean_series(raw)
    assert invalid == 1
    assert dups == 1
    assert cleaned["Close"].tolist() == [10.0, 11.0]  # first duplicate kept


# =====================================================================
# Cadence and coverage
# =====================================================================


def test_cadence_monthly_series() -> None:
    dates = pd.DatetimeIndex(pd.date_range("2000-01-31", periods=24, freq=pd.offsets.MonthEnd()))
    cadence, median = classify_cadence(dates)
    assert cadence == "monthly"
    assert 28 <= median <= 31


def test_cadence_daily_series() -> None:
    dates = pd.DatetimeIndex(pd.bdate_range("2024-08-15", periods=60))
    cadence, _ = classify_cadence(dates)
    assert cadence == "daily"


def test_coverage_none_when_no_observations_in_window() -> None:
    dates = pd.DatetimeIndex(pd.date_range("2024-08-31", "2025-03-31", freq=pd.offsets.MonthEnd()))
    status, n = assess_coverage(dates, "2023-01-01", "2023-06-30", 30.0)
    assert status == "none"
    assert n == 0


def test_coverage_partial_when_series_starts_after_window() -> None:
    dates = pd.DatetimeIndex(pd.date_range("2021-02-28", "2024-12-31", freq=pd.offsets.MonthEnd()))
    status, n = assess_coverage(dates, "2020-01-01", "2021-06-30", 30.0)
    assert status == "partial"
    assert n > 0


def test_coverage_full_when_series_spans_window() -> None:
    dates = pd.DatetimeIndex(pd.date_range("1994-05-31", "2008-12-31", freq=pd.offsets.MonthEnd()))
    status, _ = assess_coverage(dates, "2007-01-01", "2008-12-31", 30.0)
    assert status == "full"


# =====================================================================
# Entity processing and output
# =====================================================================


def test_process_entity_skips_when_event_not_covered(tmp_path) -> None:
    html = tmp_path / "SBNY_daily.html"
    html.write_text(_html(_monthly("2024-08-31", "2025-03-31")), encoding="utf-8")

    report, patch = process_entity(_entity("SBNY"), html)
    assert report.status == "skipped"
    assert report.coverage == "none"
    assert patch is None


def test_process_entity_writes_patch_with_window_flag(tmp_path) -> None:
    html = tmp_path / "LEHMQ_daily.html"
    html.write_text(_html(_monthly("2006-06-30", "2009-06-30")), encoding="utf-8")

    report, patch = process_entity(_entity("LEHMQ"), html)
    assert report.status == "written"
    assert report.cadence == "monthly"
    assert report.coverage == "full"
    assert patch is not None
    assert set(patch.columns) == {
        "Date", "Ticker", "Name", "Category", "Sector",
        "Close", "Adj_Close", "InEventWindow",
    }
    # No fabricated OHLC or Volume columns
    assert not {"Open", "High", "Low", "Volume"} & set(patch.columns)

    window = patch[patch["InEventWindow"]]
    assert window["Date"].min() >= "2007-01-01"
    assert window["Date"].max() <= "2008-12-31"


def test_extract_writes_covered_and_manifests_all(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "patches"
    raw_dir.mkdir()

    (raw_dir / "LEHMQ_daily.html").write_text(
        _html(_monthly("2006-06-30", "2009-06-30")), encoding="utf-8"
    )
    (raw_dir / "CHK_daily.html").write_text(
        _html(_monthly("2021-02-28", "2024-12-31")), encoding="utf-8"
    )

    written = extract_distressed_patches(raw_dir=raw_dir, output_dir=out_dir)

    assert set(written) == {"LEHMQ"}
    assert (out_dir / "LEHMQ.csv").exists()
    assert not (out_dir / "CHK.csv").exists()

    manifest = pd.read_csv(out_dir / "manifest.csv")
    assert len(manifest) == len(TARGET_ENTITIES)
    by_ticker = manifest.set_index("ticker")
    assert by_ticker.loc["LEHMQ", "status"] == "written"
    assert by_ticker.loc["CHK", "status"] == "skipped"
    assert by_ticker.loc["CHK", "coverage"] == "partial"
    assert by_ticker.loc["SIVB", "status"] == "skipped"  # file not provided
    assert "not found" in by_ticker.loc["SIVB", "reason"]
