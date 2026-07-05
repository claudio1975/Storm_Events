# Storm_Events

Data source: [NOAA NCEI Storm Events Database](https://www.ncei.noaa.gov/stormevents/ftp.jsp)

- Navigate to **HTTP access** to download the CSV files (compressed in `.gz` format).

## Pipeline

- **`extraction.py`** — extracts the individual CSV files from the downloaded `.gz` archives.
- **`merge.py`** — merges the extracted CSV files into a single raw dataset spanning 1950–2025.
- **`StormEvents_Cleaning_used.ipynb`** — cleans the raw dataset (fills missing values, drops unused columns) and produces three dataframes:
  - a tabular-only dataframe
  - a dataframe with both tabular and text data rows
  - a cleaned dataframe, used in all subsequent processing steps
- **`StormEvents_filling_text_generation_used.ipynb`** and **`StormEvents_residual_text_generation_used.ipynb`** — fill the two free-text columns in the cleaned dataset via text generation.