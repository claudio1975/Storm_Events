# Storm_Events

An end-to-end data pipeline for the [NOAA NCEI Storm Events Database](https://www.ncei.noaa.gov/stormevents/ftp.jsp), covering U.S. storm records from **1950 to 2025** (~1.8 million events). The pipeline merges and cleans the raw records, fills the missing narrative texts with LLM-generated ones, and enriches every event with embedding-based and LLM-derived features, producing a dataset ready for machine-learning work.

## Data source

Storm event records are published by NOAA's National Centers for Environmental Information:

1. Open the [Storm Events Database FTP page](https://www.ncei.noaa.gov/stormevents/ftp.jsp).
2. Navigate to **HTTP access** and download the yearly CSV files (compressed in `.gz` format).

## Pipeline

The scripts and notebooks are numbered in the order they run.

1. **`1_extraction.py`** Extracts the individual CSV files from the downloaded `.gz` archives.
2. **`2_merge.py`** Merges the extracted CSV files, year by year, into a single raw dataset spanning 1950–2025.
3. **`3_StormEvents_missing_values_assessment.ipynb`** Assesses the missing values in the raw merged dataset: which columns have gaps and how the missingness is distributed over time. A diagnostic step that informs the cleaning choices made next.
4. **`4_StormEvents_Cleaning_used.ipynb`** Cleans the raw dataset: fills missing values, groups the 56 raw event types into broader categories, and drops unused columns. Its output is the cleaned dataset used in all subsequent steps.
5. **`5_StormEvents_filling_text_generation_used.ipynb`** Identifies every event missing an `EPISODE_NARRATIVE` or `EVENT_NARRATIVE` and generates the missing text with OpenAI's **`gpt-4o-mini`** through the **Batch API**. Rows that already have both narratives are never sent, and when one narrative exists it is passed to the model as context so the generated text stays consistent with it.
6. **`6_StormEvents_residual_text_generation_used.ipynb`** Fills the few rows the batch jobs left behind using synchronous calls to the same `gpt-4o-mini` model, runs sanity checks, and saves the fully narrative-complete dataset.
7. **`7_StormEvents_embedding_augmentation_used.ipynb`** Encodes the episode narratives with the **`all-MiniLM-L12-v2`** sentence-transformer model and applies dimensionality reduction, turning the free text into a compact set of numeric embedding features usable alongside the tabular columns. Embeddings could also be produced through the OpenAI API, GPT-family models build internal text representations, and OpenAI exposes them through dedicated embedding models, but `gpt-4o-mini` itself is used as a generative endpoint that returns text, not vectors. A local sentence-transformer was preferred here because it is purpose-built for sentence-level embeddings, producing a fixed-length numeric vector that captures each narrative's meaning, exactly the form needed for machine-learning features. It also runs locally on the GPU, so encoding ~1.8 million narratives is fast and costs nothing in API calls.

   One could object that OpenAI's embedding models return richer vectors (1,536 dimensions or more, against MiniLM's 384) and should therefore capture more nuance. In this pipeline that advantage would be lost: the embeddings are not used raw but compressed by TruncatedSVD down to **10 components per narrative** (`ep_embedding_1..10`, `ev_embedding_1..10`), so both models funnel into the same small feature set and the extra dimensions would mostly be discarded. Vector size is also not a quality measure in itself. `all-MiniLM-L12-v2` scores strongly on sentence-similarity benchmarks despite its compact size, and the storm narratives are short weather descriptions that a compact model represents well. For this use case the larger vectors would add API cost and processing time without a measurable gain in the final 10 features.
8. **`8_StormEvents_feature_augmentation_used.ipynb`** Asks `gpt-4o-mini` to read each `EPISODE_NARRATIVE` and answer three questions, adding one categorical column per answer:

   | New column | Question the LLM answers | Possible answers |
   |---|---|---|
   | `risk` | How dangerous was the episode? | high / medium / low |
   | `impact_type` | What was mainly affected? | casualties / property_damage / crop_damage / infrastructure_disruption / no_significant_impact |
   | `event_scope` | How large an area was affected? | localized / county-wide / regional / widespread |

9. **`9_export_github.py`** Splits the datasets into compressed Parquet parts small enough for GitHub and writes them to the `data/` folder.

## Data files

The `data/` folder contains two datasets, each split into Parquet parts (zstd-compressed) to respect GitHub's file-size limits:

- **`StormEvents_part_1..2.parquet`** The raw merged dataset (output of step 2).
- **`StormEvents_fe_ep_augmentation_fin_part_1..5.parquet`** The final dataset with generated narratives, embedding features, and the three LLM-derived columns (output of step 8).

To reassemble a dataset, concatenate its parts in order:

```python
import glob
import pandas as pd

parts = sorted(glob.glob("data/StormEvents_fe_ep_augmentation_fin_part_*.parquet"))
df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
```
