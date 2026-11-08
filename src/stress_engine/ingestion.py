"""Historical distressed-entity price ingestion and normalization.

Raw HTML scrapes contain embedded {"d": <epoch seconds>, "v": <price>} records.
This module extracts them, classifies each series' sampling cadence from the data
(most sources are monthly, so cadence is never assumed), checks coverage against
each entity's event window, and writes normalized CSV patches plus a manifest.

A patch is written only when the source covers its event window. Entities whose
source does not cover the event are recorded in the manifest as skipped.
"""

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class TargetEntity:
    ticker: str
    filename: str
    name: str
    category: str
    sector: str
    event_start: str  # inclusive, YYYY-MM-DD
    event_end: str  # inclusive, YYYY-MM-DD
    event: str  # what the window is meant to capture


TARGET_ENTITIES: tuple[TargetEntity, ...] = (
    TargetEntity(
        "LEHMQ",
        "LEHMQ_daily.html",
        "Lehman Brothers",
        "Distressed_2008",
        "Financials",
        "2007-01-01",
        "2008-12-31",
        "Subprime losses through Chapter 11 filing on 2008-09-15",
    ),
    TargetEntity(
        "BSC",
        "BSC_daily.html",
        "Bear Stearns",
        "Distressed_2008",
        "Financials",
        "2007-01-01",
        "2008-06-30",
        "Mid-2007 hedge fund collapses through the March 2008 rescue and May 2008 delisting",
    ),
    TargetEntity(
        "SIVB",
        "SVB_daily.html",
        "Silicon Valley Bank",
        "Distressed_2023",
        "Financials",
        "2023-01-01",
        "2023-05-31",
        "Deposit run on 2023-03-09 and closure by regulators on 2023-03-10",
    ),
    TargetEntity(
        "SBNY",
        "SBNY_daily.html",
        "Signature Bank",
        "Distressed_2023",
        "Financials",
        "2023-01-01",
        "2023-06-30",
        "Closure by regulators on 2023-03-12",
    ),
    TargetEntity(
        "PACW",
        "PACW_daily.html",
        "PacWest Bancorp",
        "Distressed_2023",
        "Financials",
        "2023-01-01",
        "2023-12-31",
        "Regional bank stress following the SVB closure in March 2023",
    ),
    TargetEntity(
        "CHK",
        "CHK_daily.html",
        "Chesapeake Energy",
        "Distressed_2020",
        "Energy",
        "2020-01-01",
        "2021-06-30",
        "Chapter 11 filing on 2020-06-28 through emergence from bankruptcy",
    ),
)

# Matches {"d": <epoch seconds>, "v": <price>} with an optional quote around the price.
# The price must be a plain decimal number; anything else is not treated as a record.
EXTRACTION_PATTERN: re.Pattern[str] = re.compile(
    r'\{\s*"d"\s*:\s*(\d+)\s*,\s*"v"\s*:\s*"?(\d+(?:\.\d+)?)"?\s*\}'
)


@dataclass(frozen=True, slots=True)
class SeriesReport:
    ticker: str
    source_file: str
    status: str  # "written" or "skipped"
    reason: str
    cadence: str  # "daily", "weekly", "monthly", "irregular", or "n/a"
    median_gap_days: float | None
    raw_records: int
    invalid_dropped: int
    duplicate_dates_dropped: int
    source_first: str
    source_last: str
    event_start: str
    event_end: str
    coverage: str  # "full", "partial", "none", or "n/a"
    obs_in_window: int


def get_project_root() -> Path:
    """Returns absolute path to the repository root directory."""
    return Path(__file__).resolve().parents[2]


def parse_raw_records(content: str) -> pd.DataFrame:
    """Extracts (Date, Close) records from raw HTML text. Dates are normalized to midnight."""
    matches = EXTRACTION_PATTERN.findall(content)
    if not matches:
        return pd.DataFrame(columns=["Date", "Close"])

    parsed = np.array(matches, dtype=np.str_)
    dates = pd.to_datetime(parsed[:, 0].astype(np.int64), unit="s").normalize()
    prices = parsed[:, 1].astype(np.float64)
    return pd.DataFrame({"Date": dates, "Close": prices})


def clean_series(raw: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Drops non-positive or missing prices and duplicate dates.

    Returns the cleaned series and the counts of invalid rows and duplicate dates dropped.
    The first occurrence of a duplicate date is kept.
    """
    invalid_mask = ~(raw["Close"] > 0.0)
    invalid_dropped = int(invalid_mask.sum())
    valid = raw.loc[~invalid_mask].sort_values("Date")

    duplicate_mask = valid.duplicated(subset=["Date"], keep="first")
    duplicate_dropped = int(duplicate_mask.sum())
    cleaned = valid.loc[~duplicate_mask].reset_index(drop=True)
    return cleaned, invalid_dropped, duplicate_dropped


def classify_cadence(dates: pd.DatetimeIndex) -> tuple[str, float | None]:
    """Classifies sampling cadence from the median gap between observations."""
    if len(dates) < 2:
        return "n/a", None

    gaps_days = np.diff(dates.values) / np.timedelta64(1, "D")
    median_gap = float(np.median(gaps_days))

    if median_gap <= 3:
        cadence = "daily"
    elif 5 <= median_gap <= 9:
        cadence = "weekly"
    elif 25 <= median_gap <= 35:
        cadence = "monthly"
    else:
        cadence = "irregular"
    return cadence, median_gap


def assess_coverage(
    dates: pd.DatetimeIndex,
    event_start: str,
    event_end: str,
    median_gap: float | None,
) -> tuple[str, int]:
    """Reports whether the series covers the event window.

    "full" means observations exist near both ends of the window, allowing for the
    sampling gap. "partial" means some observations fall inside the window but not
    at both ends. "none" means no observations fall inside the window.
    """
    start_ts = pd.Timestamp(event_start)
    end_ts = pd.Timestamp(event_end)

    obs_in_window = int(((dates >= start_ts) & (dates <= end_ts)).sum())
    if obs_in_window == 0:
        return "none", 0

    gap = median_gap if median_gap is not None else 31.0
    tolerance = pd.Timedelta(days=max(7.0, 1.5 * gap))
    if dates.min() <= start_ts + tolerance and dates.max() >= end_ts - tolerance:
        return "full", obs_in_window
    return "partial", obs_in_window


def process_entity(
    entity: TargetEntity, file_path: Path
) -> tuple[SeriesReport, pd.DataFrame | None]:
    """Parses one source file and decides whether it is written as a patch."""
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    raw = parse_raw_records(content)
    raw_records = len(raw)

    base = dict(
        ticker=entity.ticker,
        source_file=file_path.name,
        raw_records=raw_records,
        event_start=entity.event_start,
        event_end=entity.event_end,
    )

    if raw_records == 0:
        return (
            SeriesReport(
                **base,
                status="skipped",
                reason="No price records matched the extraction pattern",
                cadence="n/a",
                median_gap_days=None,
                invalid_dropped=0,
                duplicate_dates_dropped=0,
                source_first="",
                source_last="",
                coverage="n/a",
                obs_in_window=0,
            ),
            None,
        )

    cleaned, invalid_dropped, duplicate_dropped = clean_series(raw)
    if cleaned.empty:
        return (
            SeriesReport(
                **base,
                status="skipped",
                reason="No valid positive prices after cleaning",
                cadence="n/a",
                median_gap_days=None,
                invalid_dropped=invalid_dropped,
                duplicate_dates_dropped=duplicate_dropped,
                source_first="",
                source_last="",
                coverage="n/a",
                obs_in_window=0,
            ),
            None,
        )

    dates = pd.DatetimeIndex(cleaned["Date"])
    cadence, median_gap = classify_cadence(dates)
    coverage, obs_in_window = assess_coverage(
        dates, entity.event_start, entity.event_end, median_gap
    )

    shared = dict(
        base,
        cadence=cadence,
        median_gap_days=median_gap,
        invalid_dropped=invalid_dropped,
        duplicate_dates_dropped=duplicate_dropped,
        source_first=dates.min().strftime("%Y-%m-%d"),
        source_last=dates.max().strftime("%Y-%m-%d"),
        coverage=coverage,
        obs_in_window=obs_in_window,
    )

    if coverage != "full":
        reason = (
            f"Source does not cover the event window ({entity.event}); "
            f"coverage is '{coverage}' with {obs_in_window} observations in window"
        )
        return SeriesReport(**shared, status="skipped", reason=reason), None

    in_window = (dates >= pd.Timestamp(entity.event_start)) & (
        dates <= pd.Timestamp(entity.event_end)
    )
    patch = pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Ticker": entity.ticker,
            "Name": entity.name,
            "Category": entity.category,
            "Sector": entity.sector,
            "Close": cleaned["Close"].to_numpy(),
            # Source adjustment status is unverified; Adj_Close mirrors Close until checked
            "Adj_Close": cleaned["Close"].to_numpy(),
            "InEventWindow": in_window.to_numpy(),
        }
    )
    reason = f"Event window covered; written with cadence '{cadence}'"
    return SeriesReport(**shared, status="written", reason=reason), patch


def extract_distressed_patches(
    raw_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Parses raw HTML scrapes, writes covered patches, and writes manifest.csv.

    The manifest lists every entity, including skipped ones, with the reason.
    Returns the patch paths for entities that were written.
    """
    root = get_project_root()
    source_dir = raw_dir if raw_dir is not None else root / "data" / "raw"
    target_dir = output_dir if output_dir is not None else root / "data" / "patches"
    target_dir.mkdir(parents=True, exist_ok=True)

    generated_patches: dict[str, Path] = {}
    reports: list[SeriesReport] = []

    for entity in TARGET_ENTITIES:
        file_path = source_dir / entity.filename

        # Fallback for the SIVB ticker, whose source file is named SVB_daily.html
        if not file_path.exists() and entity.ticker == "SIVB":
            fallback = source_dir / "SIVB_daily.html"
            if fallback.exists():
                file_path = fallback

        if not file_path.exists():
            report = SeriesReport(
                ticker=entity.ticker,
                source_file=file_path.name,
                status="skipped",
                reason=f"Source file not found: {file_path}",
                cadence="n/a",
                median_gap_days=None,
                raw_records=0,
                invalid_dropped=0,
                duplicate_dates_dropped=0,
                source_first="",
                source_last="",
                event_start=entity.event_start,
                event_end=entity.event_end,
                coverage="n/a",
                obs_in_window=0,
            )
            reports.append(report)
            print(f"[-] Skipped {entity.ticker}: {report.reason}")
            continue

        report, patch = process_entity(entity, file_path)
        reports.append(report)

        if patch is None:
            print(f"[-] Skipped {entity.ticker}: {report.reason}")
            continue

        dest_csv = target_dir / f"{entity.ticker}.csv"
        patch.to_csv(dest_csv, index=False)
        generated_patches[entity.ticker] = dest_csv
        print(
            f"[+] Written {entity.ticker} -> {dest_csv.name} "
            f"[{report.cadence}, {report.raw_records} records, "
            f"{report.obs_in_window} in window]"
        )

    manifest_path = target_dir / "manifest.csv"
    pd.DataFrame([asdict(r) for r in reports]).to_csv(manifest_path, index=False)
    print(f"[Manifest] {manifest_path}")

    return generated_patches


if __name__ == "__main__":
    extract_distressed_patches()
