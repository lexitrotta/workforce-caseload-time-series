# scripts/caseload_data_prep.py

import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/caseload_raw.csv")
PROC_PATH = Path("data/processed/caseload_monthly.csv")

def main():
    df = pd.read_csv(RAW_PATH, parse_dates=["date"])
    
    # Example: aggregate to monthly caseload
    df["year_month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        df.groupby("year_month")
        .agg(total_cases=("case_id", "nunique"))
        .reset_index()
        .sort_values("year_month")
    )

    PROC_PATH.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(PROC_PATH, index=False)
    print(f"Saved monthly caseload data to {PROC_PATH}")

if __name__ == "__main__":
    main()
