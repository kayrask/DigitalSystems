#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import re
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
FORMULAS_DIR = BASE_DIR / "data" / "formulas"

def find_col(df, keywords):
    # return first column whose name contains any keyword
    for c in df.columns:
        cl = str(c).strip().lower()
        for kw in keywords:
            if kw in cl:
                return c
    return None

def to_float(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    s = s.replace("%", "").replace(",", ".")
    try:
        return float(s)
    except:
        return None

def is_probably_table(df):
    # heuristic: needs at least 3 columns and 5 rows
    return df.shape[1] >= 3 and df.shape[0] >= 5

def inspect_excel(path: Path):
    print("\n" + "="*80)
    print(f"FILE: {path.relative_to(BASE_DIR)}")
    print("="*80)

    xl = pd.ExcelFile(path)
    for sheet in xl.sheet_names:
        try:
            df = xl.parse(sheet)
        except Exception as e:
            print(f"\n  [Sheet: {sheet}] -> FAILED to parse: {e}")
            continue

        if df is None or df.empty or not is_probably_table(df):
            continue

        # try to detect INCI and percent columns
        inci_col = find_col(df, ["inci"])
        pct_col  = find_col(df, ["%", "yüzde", "yuzde", "percent"])
        name_col = find_col(df, ["hammadde", "ingredient", "madde", "name"])

        if inci_col is None and name_col is None:
            continue  # not a formula sheet

        print(f"\n  [Sheet: {sheet}]")
        print(f"    detected inci_col={inci_col} | name_col={name_col} | pct_col={pct_col}")

        rows = []
        for _, r in df.iterrows():
            inci = str(r[inci_col]).strip() if inci_col and r.get(inci_col) is not None else ""
            name = str(r[name_col]).strip() if name_col and r.get(name_col) is not None else ""
            pct  = to_float(r[pct_col]) if pct_col else None

            val = inci or name
            if not val or val.lower() == "nan":
                continue

            # skip obvious phase headers
            if val.lower() in ["faz", "phase", "a", "b", "c", "d"]:
                continue

            rows.append((val, pct))

        if not rows:
            print("    (no ingredient rows detected)")
            continue

        # Print a clean list
        print("    Ingredients:")
        for val, pct in rows:
            if pct is None:
                print(f"      - {val}")
            else:
                print(f"      - {val}  |  {pct}%")

def main():
    if not FORMULAS_DIR.exists():
        print(f"ERROR: formulas folder not found: {FORMULAS_DIR}")
        print("Make sure your formulas are under: data/formulas/")
        sys.exit(1)

    files = sorted(list(FORMULAS_DIR.rglob("*.xlsx")))
    if not files:
        print(f"ERROR: no .xlsx found under {FORMULAS_DIR}")
        sys.exit(1)

    print(f"Found {len(files)} Excel file(s) under {FORMULAS_DIR}")
    for f in files:
        inspect_excel(f)

if __name__ == "__main__":
    main()
