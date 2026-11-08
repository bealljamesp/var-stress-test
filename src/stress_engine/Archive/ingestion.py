"""Historical distressed entity data extraction and schema normalization."""

import re
from dataclasses import dataclass
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


TARGET_ENTITIES: tuple[TargetEntity, ...] = (
    TargetEntity(
        "LEHMQ", "LEHMQ_daily.html", "Lehman Brothers", "Distressed_2008", "Financials"
    ),
    TargetEntity(
        "BSC", "BSC_daily.html", "Bear Stearns", "Distressed_2008", "Financials"
    ),
    TargetEntity(
        "SIVB", "SVB_daily.html", "Silicon Valley Bank", "Distressed_2023", "Financials"
    ),
    TargetEntity(
        "SBNY", "SBNY_daily.html", "Signature Bank", "Distressed_2023", "Financials"
    ),
    TargetEntity(
        "PACW", "PACW_daily.html", "PacWest Bancorp", "Distressed_2023", "Financials"
    ),
    TargetEntity(
        "CHK", "CHK_daily.html", "Chesapeake Energy", "Distressed_2020", "Energy"
    ),
)

# Regex pattern matching {"d": <epoch>, "v": <price>}
EXTRACTION_PATTERN: re.Pattern[str] = re.compile(
    r'{\s*"d"\s*:\s*(\d+)\s*,\s*"v"\s*:\s*"?([\d.]+)"?\s*}'
)


def get_project_root() -> Path:
    """Returns absolute path to the repository root directory."""
    return Path(__file__).resolve().parents[2]


def parse_entity_html(html_path: Path) -> pd.DataFrame:
    """Extracts date and price arrays using vectorized parsing from raw HTML text."""
    content = html_path.read_text(encoding="utf-8", errors="ignore")
    matches = EXTRACTION_PATTERN.findall(content)

    if not matches:
        return pd.DataFrame(columns=["Date", "Close", "Adj_Close"])

    # Vectorized conversion via 2D string array to avoid row-by-row dict overhead
    parsed_array = np.array(matches, dtype=np.str_)
    dates = pd.to_datetime(parsed_array[:, 0].astype(np.int64), unit="s")
    prices = parsed_array[:, 1].astype(np.float64)

    df = pd.DataFrame(
        {
            "Date": dates,
            "Close": prices,
            "Adj_Close": prices,
        }
    )
    return (
        df.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
    )


def extract_distressed_patches(
    raw_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Parses local raw HTML scrapings and writes normalized CSV patches."""
    root = get_project_root()
    source_dir = raw_dir if raw_dir is not None else root / "data" / "raw"
    target_dir = output_dir if output_dir is not None else root / "data" / "patches"
    target_dir.mkdir(parents=True, exist_ok=True)

    generated_patches: dict[str, Path] = {}

    for entity in TARGET_ENTITIES:
        file_path = source_dir / entity.filename

        # Fallback resolution for naming discrepancies (e.g. SIVB vs SVB)
        if not file_path.exists() and entity.ticker == "SIVB":
            fallback = source_dir / "SIVB_daily.html"
            if fallback.exists():
                file_path = fallback

        if not file_path.exists():
            print(f"[-] Skipped: File not found: {file_path}")
            continue

        df_entity = parse_entity_html(file_path)
        if df_entity.empty:
            print(f"[!] Warning: No price pattern matches found in: {file_path.name}")
            continue

        # Broadcast metadata across the contiguous dataframe
        df_entity["Ticker"] = entity.ticker
        df_entity["Name"] = entity.name
        df_entity["Category"] = entity.category
        df_entity["Sector"] = entity.sector
        df_entity["Open"] = df_entity["Close"]
        df_entity["High"] = df_entity["Close"]
        df_entity["Low"] = df_entity["Close"]
        df_entity["Volume"] = 0

        # Canonical institutional column schema
        canonical_columns = [
            "Date",
            "Ticker",
            "Name",
            "Category",
            "Sector",
            "Open",
            "High",
            "Low",
            "Close",
            "Adj_Close",
            "Volume",
        ]
        df_canonical = df_entity.loc[:, canonical_columns]

        dest_csv = target_dir / f"{entity.ticker}.csv"
        df_canonical.to_csv(dest_csv, index=False)
        generated_patches[entity.ticker] = dest_csv
        print(
            f"[+] Processed: {entity.name} ({entity.ticker}) -> {dest_csv.name} [{len(df_canonical):,} rows]"
        )

    return generated_patches


if __name__ == "__main__":
    extract_distressed_patches()
