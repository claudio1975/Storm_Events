import pandas as pd
import glob
import os
import re

# ── CONFIGURATION ─────────────────────────────────────────────
INPUT_FOLDER  = r"./data"
OUTPUT_FOLDER = r"./work"
OUTPUT_FILE   = "StormEvents.csv"
YEAR_PATTERN  = r"_d(\d{4})_"   
# ──────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def extract_year(filename):
    match = re.search(YEAR_PATTERN, filename)
    return match.group(1) if match else None

# Step 1 — merge per year
csv_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.csv")))
print(f"Trovati {len(csv_files)} file CSV\n")

files_by_year = {}
for file in csv_files:
    year = extract_year(os.path.basename(file))
    if year:
        files_by_year.setdefault(year, []).append(file)

yearly_dfs = []
for year, files in sorted(files_by_year.items()):
    dfs = [pd.read_csv(f, low_memory=False) for f in files]
    merged_year = pd.concat(dfs, ignore_index=True)
    yearly_dfs.append(merged_year)
    print(f"  ✓ {year}  —  {len(files)} file  —  {len(merged_year):,} righe")

# Step 2 — one file
final = pd.concat(yearly_dfs, ignore_index=True)
out_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)
final.to_csv(out_path, index=False)

