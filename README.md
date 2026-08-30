# Storm_Events

An end-to-end data pipeline for the [NOAA NCEI Storm Events Database](https://www.ncei.noaa.gov/stormevents/ftp.jsp), covering U.S. storm records from **1950 to 2025** (more than 2 million events). The pipeline merges and cleans the raw records, fills the missing narrative texts with LLM-generated ones, and enriches every event with embedding-based and LLM-derived features, producing a dataset ready for machine-learning work. 

## Data source

Storm event records are published by NOAA's National Centers for Environmental Information:

1. Open the [Storm Events Database FTP page](https://www.ncei.noaa.gov/stormevents/ftp.jsp).
2. Navigate to **HTTP access** and download the yearly CSV files (compressed in `.gz` format).

### What the source period of record actually means

![NOAA Storm Events Database period of record](images/Storm_DataBase_info.jpg)

This is NOAA's own description of the database, and it is the single most important thing to understand before modelling anything. The database spans 1950 to today, but **it is not one homogeneous series**, it is three different data-collection regimes stacked end to end:

- **1950–1954**: only **tornado** events were recorded.
- **1955–1995**: **tornado, thunderstorm wind and hail**.
- **1996–present**: all **48 event types** defined in [NWS Directive 10-1605](https://www.ncei.noaa.gov/stormevents/ftp.jsp) are recorded.

## Pipeline

The scripts and notebooks are numbered in the order they run.

1. **`1_extraction.py`** Extracts the individual CSV files from the downloaded `.gz` archives.
2. **`2_merge.py`** Merges the extracted CSV files, year by year, into a single raw dataset spanning 1950–2025.
3. **`3_StormEvents_Cleaning_used.ipynb`** Assesses the missing values (which columns have gaps and how the missingness is distributed over time), then cleans the raw dataset: parses the damage strings, normalises the tornado intensity scale, groups the 56 raw event types into broader **event groups**, fills missing values, and drops columns that are unpopulated or redundant. Its output is the cleaned dataset used in all subsequent steps.
4. **`4_StormEvents_filling_text_generation_used.py`** Identifies every event missing an `EPISODE_NARRATIVE` or `EVENT_NARRATIVE` and generates the missing text with OpenAI's **`gpt-4o-mini`** through the **Batch API**. Rows that already have both narratives are never sent, and when one narrative exists it is passed to the model as context so the generated text stays consistent with it. The structured fields of the row (event type, state, dates, magnitude, damages, casualties…) are supplied as grounding context so the generated narrative is factual rather than invented.
5. **`5_StormEvents_embedding_augmentation_used.py`** Encodes the episode and event narratives with the **`all-MiniLM-L12-v2`** sentence-transformer model and applies dimensionality reduction, turning the free text into a compact set of numeric embedding features usable alongside the tabular columns. Embeddings could also be produced through the OpenAI API — GPT-family models, but `gpt-4o-mini` itself is used as a generative endpoint that returns text, not vectors. A local sentence-transformer was preferred here because it is purpose-built for sentence-level embeddings, producing a fixed-length numeric vector that captures each narrative's meaning, exactly the form needed for machine-learning features. 

   One could object that OpenAI's embedding models return richer vectors (1,536 dimensions or more, against MiniLM's 384) and should therefore capture more nuance. In this pipeline that advantage would be lost: the embeddings are not used raw but compressed by TruncatedSVD down to **10 components per narrative** (`ep_embedding_1..10`, `ev_embedding_1..10`), so both models funnel into the same small feature set and the extra dimensions would mostly be discarded. Vector size is also not a quality measure in itself. `all-MiniLM-L12-v2` scores strongly on sentence-similarity benchmarks despite its compact size, and the storm narratives are short weather descriptions that a compact model represents well. For this use case the larger vectors would add API cost and processing time without a measurable gain in the final 10 features.
6. **`6_StormEvents_feature_augmentation_used.py`** Asks `gpt-4o-mini`, again through the Batch API, to read each `EPISODE_NARRATIVE` and answer three questions, adding one categorical column per answer:

   | New column | Question the LLM answers | Possible answers |
   |---|---|---|
   | `risk` | How dangerous was the episode? | high / medium / low |
   | `impact_type` | What was mainly affected? | casualties / property_damage / crop_damage / infrastructure_disruption / no_significant_impact |
   | `event_scope` | How large an area was affected? | localized / county-wide / regional / widespread |

   Each label is defined explicitly in the system prompt (e.g. `high` = deaths, injuries or major destruction occurred or were clearly likely) so the classification stays consistent across the whole dataset, and the model is instructed to judge only what the text states rather than assume unmentioned impacts.
7. **`7_export_github.py`** Splits the datasets into compressed Parquet parts small enough for GitHub and writes them to the `data/` folder.
8. **`8_EDA_used.ipynb`** Exploratory analysis of the final dataset: coverage over time, event-group composition, geography, damage and casualty trends, stationarity and seasonality of the monthly series, and the categorical drivers of impact. Some pictures below come from this notebook.
9. **`9_Global_Prediction_used_count.ipynb`**, **`9_Thunderstorm_prediction_used_count.ipynb`**, **`9_Tornado_prediction_used_count.ipynb`** Three parallel modelling notebooks that share one design and differ only in the slice of the database they cover. The split follows the collection-regime problem described above: Tornado and Thunderstorm have event records long before 1996, so they are modelled separately, while the remaining event groups are modelled together from 1996 onward.

   | Notebook | Scope | Period | Events 
   |---|---|---|---|
   | `9_Global_…` | the 11 event groups left after removing Tornado, Thunderstorm and the four almost-empty groups (`Geomagnetic`, `Volcanic`, `Tsunami`, `Marine_Other`) | 1996–2025 | 888,673 
   | `9_Thunderstorm_…` | `Thunderstorm` | 1955–2025 | 1,032,841 
   | `9_Tornado_…` | `Tornado` | 1950–2025 | 90,255 

   All three read the augmented dataset straight from the parquet files in `data/` and follow the same protocol. The data is **split by date first** (train through 2023, calibration 2024, test 2025) and only then filtered for zero-variance, redundant and high-cardinality features **using training rows alone**, so no test information reaches the design matrix. The GLM, LightGBM and LSTM hyperparameters are selected with expanding-window cross-validation over validation years, minimizing mean Poisson deviance. TabPFN-3 uses a fixed in-context configuration with a target-stratified context of up to 50,000 training rows, prioritising positive casualty cases and sampling zeros to fill the context. All four models are then backtested on the same expanding-window folds. Every model is backtested on the same expanding-window folds it was tuned on and produces 90% adaptive conformal intervals. Some pictures below come from these notebooks.

## What the data looks like

### Events per year

![Storm events per year](images/Storm_Events_x_Year.png)

The yearly event count is flat and low (a few thousand per year) through the 1950s–1980s, then rises through the early 1990s and **jumps almost fourfold between 1995 and 1996**, from roughly 9,000 to nearly 50,000 events. That vertical wall is not a climate signal: it is the 1996 switch to recording all 48 event types described above. After 1996 the series settles into a genuine range of roughly 50,000–80,000 events per year, with the peaks (2008, 2011, 2023–2025) reflecting real, severe storm seasons.

### Composition by event group

![Rows per event group with first and last recorded year](images/Storm_Events_Granularity.png)

The 56 raw NOAA event types are collapsed into 17 broader **event groups**. The chart shows how many rows each group holds and the first/last year it appears, and it makes two things immediately clear.

First, the distribution is extremely **imbalanced**. `Thunderstorm` alone accounts for about 1.03 million rows, more than half the dataset. At the other end, `Geomagnetic` has 8 rows, `Tsunami` 52 and `Volcanic` 147. 

Second, the **start years confirm the collection-regime story**: `Tornado` starts in 1950 and `Thunderstorm` in 1955, while nearly every other group starts in exactly 1996. A few start even later simply because the phenomenon is rare or was catalogued later (`Marine_Other` 2002, `Tsunami` 2006).

### Geographic distribution

![Storm event locations, 60,000 sampled points](images/Storm_Event_Location.png)

A 60,000-point sample of event coordinates over the continental U.S. The density map matches known U.S. severe-weather climatology: a dense core across the Great Plains and the Midwest into the Southeast (Tornado Alley and Dixie Alley), heavy coverage along the Gulf and Atlantic coasts and the Florida peninsula, and a comparatively sparse, clustered West where events concentrate around populated valleys and mountain corridors.

Part of that east/west contrast is meteorological and part is **reporting bias**: storm events are recorded when someone observes and reports them, so sparsely populated areas generate fewer records for the same weather. 

## Casualty-Count Modeling results

The unit of analysis is the **single storm event**: one row, one event. The four targets are its casualty counts (`INJURIES_DIRECT`, `INJURIES_INDIRECT`, `DEATHS_DIRECT`, `DEATHS_INDIRECT`), and every model is fitted with a Poisson objective, so what it returns is the **expected number of casualties for that event, conditional on the event's characteristics**. The GLM, LightGBM and LSTM use an explicit Poisson count objective/loss; TabPFN-3 instead works on a transformed casualty rate before returning predictions to the count scale.

Point predictions are produced per event. For uncertainty assessment, event-level actuals and predictions are aggregated by calendar day and the 90% adaptive conformal interval is evaluated against the daily casualty total in the model-comparison charts. 
Metrics are D2 (Poisson pseudo-R², high = better), RMSE and mean Poisson deviance (MPD, low = better).

### Global — 11 event groups, 1996–2025

![Point metrics comparison by target and split, global model](images/Global_Models_Comparison_Metrics.png)

On the test split **TabPFN-3 wins all four targets** on test (D2 0.53 deaths direct, 0.67 deaths indirect, 0.48 injuries direct, 0.47 injuries indirect) and has the lowest MPD everywhere. LightGBM is the closest competitor, the LSTM sits behind it, and the GLM is the weakest, it barely beats the intercept on `INJURIES_DIRECT` (D2 0.03).

![Conformal coverage and relative width by target, global model](images/Global_Models_Comparison_Conformal.png)

Coverage lands between 0.90 and 0.94 for every model and target, so the bands are honest rather than merely narrow. **LightGBM produces the tightest intervals** on three of the four targets; TabPFN-3 is tightest on `DEATHS_INDIRECT`.

#### Backtesting comparison — `INJURIES_DIRECT` and `DEATHS_DIRECT`

The charts above measure **point prediction on the test split**: they ask which model predicted the 2025 casualty counts best. That is one year, so it says who won once, not who can be trusted next time. **Backtesting** asks the second question: each model is re-fitted on the five expanding-window folds it was tuned on (train ≤ 2018 → predict 2019, train ≤ 2019 → predict 2020, and so on up to predict 2023) and scored on each of those five years separately. A model is reliable when it stays good in *all* of them. The two direct targets are focused here because they are the ones with enough casualty events to be read year by year.

Two things come out of it:

- **On `INJURIES_DIRECT`, LightGBM is the only model that never fails.** It is positive in all five backtest years and the best of the four models in four of them. TabPFN-3 and the LSTM each have a strong year and then a negative one, i.e. a year where they predict worse than simply using the average. TabPFN-3 does win the 2025 test split on this target, but that is a single year, and the backtest shows the win does not repeat. The conclusion is therefore the opposite of what the test split alone would suggest: on injuries direct the dependable model is LightGBM, and TabPFN-3's lead should not be generalised from one year.
- **On `DEATHS_DIRECT`, the choice of model barely matters.** LightGBM, the LSTM and TabPFN-3 are all positive in every backtest year and close to each other; the GLM is last every year but still clearly positive. What makes this target easier is not the model but the amount of signal: it records roughly three times as many events with at least one casualty per year as injuries direct, so no single episode can dominate the score.

![LightGBM backtest metrics by validation year, injuries direct, global model](images/Global_LightGBM_Backtest_Metrics_Injuries_Direct.png)

![LightGBM backtest metrics by validation year, deaths direct, global model](images/Global_LightGBM_Backtest_Metrics_Deaths_Direct.png)


#### Conformal intervals by event group — `INJURIES_DIRECT` and `DEATHS_DIRECT`

Because LightGBM is both the most stable model on the backtest and the one with the tightest bands on the two direct targets, its 2025 intervals are the ones broken down per event group below.

![LightGBM monthly conformal intervals by event group, injuries direct, 2025, global model](images/Global_LightGBM_Conformal_by_Event_Injuries_Direct.png)

The 90% adaptive conformal interval for `INJURIES_DIRECT` in the 2025 test year, broken down by event group. Frequent, well-populated groups (`Winter_Storm`, `High_Wind`, `Flooding`) get bands that track the actual monthly counts. Rare groups (`Tropical_Cyclone`, `Avalanche`) get much wider, flatter bands, and occasional one-off spikes (`Extreme_Heat`) fall outside even a 90% band entirely, a direct, visual consequence of the same data-scarcity problem that also produces the undefined D2 scores for `Drought` and `Tropical_Cyclone` seen elsewhere in the notebook.

![LightGBM monthly conformal intervals by event group, deaths direct, 2025, global model](images/Global_LightGBM_Conformal_by_Event_Deaths_Direct.png)

The same view for `DEATHS_DIRECT` is visibly calmer. The well-populated groups (`Winter_Storm`, `High_Wind`, `Coastal_Flood`, `Cold`) keep the monthly totals inside the band almost all year, and `Extreme_Heat`, the group that breaks the injuries picture, is here one of the best-tracked, its summer hump followed by the band from June through September. Two single episodes still escape: the January wildfire month, an order of magnitude above its band ceiling, and the July flooding peak, well above its own. They are exactly the two groups whose test D2 suffers, `Wildfire` being the only negative group of the eleven. The zero-casualty groups behave as before: `Drought`, `Dust` and `Tropical_Cyclone` record no deaths at all in 2025, so their bands stay wide and flat and their D2 is undefined, the interval covers the actuals trivially rather than informatively.

### Thunderstorm — 1955–2025

![Point metrics comparison by target and split, thunderstorm model](images/Thunderstorm_Models_Comparison_Metrics.png)

The largest and most homogeneous slice, and the one where the gradient booster is at its best: **LightGBM leads on all four test targets** (D2 0.43 injuries direct, 0.43 deaths direct, 0.37 injuries indirect, 0.27 deaths indirect), with TabPFN-3 statistically level on the two direct targets and the GLM roughly half as good.

![Conformal coverage and relative width by target, thunderstorm model](images/Thunderstorm_Models_Comparison_Conformal.png)

Coverage is again close to nominal for all four models. LightGBM gives the tightest bands on three targets, the LSTM on `INJURIES_INDIRECT`. Widths are relatively larger than in the global notebook because thunderstorm casualties are rare per event.

#### Backtesting comparison — `INJURIES_DIRECT` and `DEATHS_DIRECT`

The same five expanding-window folds as the global slice (train ≤ 2018 → predict 2019, up to train ≤ 2022 → predict 2023), read on the same two direct targets.

Two things come out of it:

- **On `INJURIES_DIRECT` the three nonlinear models are stable and close to interchangeable.** LightGBM, the LSTM and TabPFN-3 are positive in all five years and stay in a narrow range; TabPFN-3 is ahead in three years, LightGBM in the other two, and in every year the distance between them is small. The gap that repeats is the one against the GLM, which is roughly half as good year after year. Unlike the global slice, backtesting and the 2025 test split agree here: on a large, homogeneous slice the ranking does not depend on which year you look at.
- **On `DEATHS_DIRECT` LightGBM is the most consistent model**, best in four of the five years, and again nothing ever turns negative. The one real movement in the whole thunderstorm backtest is here: the level starts near 0.48 in 2019–2020 and drifts down to about 0.37 by 2023, for all four models at once, which makes it a property of those years rather than of any algorithm. 

![LightGBM backtest metrics by validation year, injuries direct, thunderstorm model](images/Thunderstorm_LightGBM_Backtest_Metrics_Injuries_Direct.png)

![LightGBM backtest metrics by validation year, deaths direct, thunderstorm model](images/Thunderstorm_LightGBM_Backtest_Metrics_Deaths_Direct.png)

The two panels are the visual contrast with the global slice: no year collapses, and the metrics stay on the same scale from 2019 to 2023. On `DEATHS_DIRECT` all three metrics are essentially flat. On `INJURIES_DIRECT` only RMSE moves, tracking the severity of each year's worst outbreaks (highest in 2020 and 2023), while D2 and MPD barely react, the same reason the ranking here is read on D2 and MPD.

#### Conformal intervals over the year — `INJURIES_DIRECT` and `DEATHS_DIRECT`

Thunderstorm is a single event group, so there is no per-group breakdown to make: the monthly view *is* the whole slice. LightGBM is shown again, being the tightest of the four models on `DEATHS_DIRECT` and tied tightest on `INJURIES_DIRECT` at nominal coverage.

![LightGBM monthly conformal intervals, injuries direct, 2025, thunderstorm model](images/Thunderstorm_LightGBM_Conformal_Monthly_Injuries_Direct.png)

`INJURIES_DIRECT` in the 2025 test year. The slice has a strong, clean seasonality, the season builds from February, peaks in June, decays through the autumn, and the band follows that shape instead of staying flat, widening exactly where the risk concentrates. The actual monthly total stays inside the interval in every month of the year, the June peak included. The predicted line sits below the actual at the spring and summer peaks, so the model under-calls the level while the interval still contains it, which is precisely what the conformal layer is there to provide.

![LightGBM monthly conformal intervals, deaths direct, 2025, thunderstorm model](images/Thunderstorm_LightGBM_Conformal_Monthly_Deaths_Direct.png)

`DEATHS_DIRECT` shows the same seasonal shape on a much smaller scale, and one month escapes: June, the peak of the season, with 20 deaths against a band ceiling near 17.5. Every other month is inside. With only a few dozen death-carrying events per year in this slice, a single severe June outbreak is exactly the kind of episode the interval cannot anticipate, the same scarcity effect seen in the global slice, but confined to one month here instead of scattered across the rare event groups.

### Tornado — 1950–2025

![Point metrics comparison by target and split, tornado model](images/Tornado_Models_Comparison_Metrics.png)

The hardest slice, with only 1,776 events in the 2025 test year. **TabPFN-3 is the best model on the two targets that carry enough test signal**: D2 0.40 on `DEATHS_DIRECT` and 0.22 on `INJURIES_DIRECT`, where the GLM is actually negative (−0.27), i.e. worse than predicting the mean. The deaths indirect target should be read as not evaluable rather than as results.

![Conformal coverage and relative width by target, tornado model](images/Tornado_Models_Comparison_Conformal.png)

Coverage stays at or above nominal for every model, but the intervals are are relatively larger than in the global notebook, and the relative width for `DEATHS_INDIRECT` is undefined because the actual total is zero all year.

#### Backtesting comparison — `INJURIES_DIRECT` and `DEATHS_DIRECT`

The same five expanding-window folds, on the thinnest slice of the three: each validation year holds only about 1,500–2,100 tornado events, of which 71–97 carry an injury and 14–31 a death.

Two things come out of it:

- **The 2025 test year is harder than any of the five backtest years.** On `INJURIES_DIRECT` the models score between 0.42 and 0.68 in every backtest year — LightGBM best in four of the five, TabPFN-3 in 2022, the two always close — against the 0.22 reported above on the 2025 test split. Same models, same protocol, so the modest test-split figure is a property of 2025 rather than evidence that the tornado slice resists modelling.
- **On `DEATHS_DIRECT` TabPFN-3 is the most consistent model**, best in four of the five years, with LightGBM just behind it and the LSTM behind both; the GLM is last every year and turns negative in 2022. That year is the weakest for every model on both targets, and it is also the year with the fewest casualty-carrying events. With only a few dozen of them per year, whether the season was violent or quiet moves the score more than the choice of algorithm does.

![TabPFN-3 backtest metrics by validation year, injuries direct, tornado model](images/Tornado_TabPFN_Backtest_Metrics_Injuries_Direct.png)

![TabPFN-3 backtest metrics by validation year, deaths direct, tornado model](images/Tornado_TabPFN_Backtest_Metrics_Deaths_Direct.png)

TabPFN-3 year by year on the two targets: D2 stays in a 0.42–0.67 band on injuries and a 0.28–0.64 band on deaths, with no negative year anywhere. RMSE and MPD track the size of each season instead of model quality — 2022, the quietest year, has the lowest values on both targets, and 2023 the highest MPD on injuries.

#### Conformal intervals over the year — `INJURIES_DIRECT` and `DEATHS_DIRECT`

Tornado is a single event group as well, so again the monthly view is the whole slice. TabPFN-3 is shown here: it is the best model on the 2025 test split for both direct targets, the most consistent on the deaths backtest, and it carries the tightest bands on `INJURIES_DIRECT` (relative width 2.94 against LightGBM's 3.55), essentially tied with the LSTM on `DEATHS_DIRECT`.

![TabPFN-3 monthly conformal intervals, injuries direct, 2025, tornado model](images/Tornado_TabPFN_Conformal_Monthly_Injuries_Direct.png)

`INJURIES_DIRECT` in the 2025 test year. The whole season lives in February–May and the rest of the year is essentially empty, and the band follows that: very wide in March, near zero from July on. March is also the one month that fails, and it fails in the safe direction — the model expected a major outbreak month (about 260 injuries, with a band running from roughly 140 to 380) and the season delivered 71, below the lower edge. From April onward predicted and actual converge and every month falls inside the interval.

![TabPFN-3 monthly conformal intervals, deaths direct, 2025, tornado model](images/Tornado_TabPFN_Conformal_Monthly_Deaths_Direct.png)

`DEATHS_DIRECT` misses twice, in opposite directions, and both misses are about *timing* rather than level: in March the band sits entirely above the 21 deaths actually recorded, and in May the 26 deaths recorded clear a ceiling near 15. The model placed the deadly month of the season in March; 2025 put it in May. The two errors partly cancel over the year, which is why the annual point metrics for this target stay respectable while the monthly intervals fail twice. With 14–31 death-carrying tornado events in a year, a one-month shift in the season is enough to break the calibration — the clearest illustration across the three slices of what data scarcity does to interval forecasts.

### Takeaways

- **No single model wins everywhere.** TabPFN-3 is the strongest on the two sparse, heterogeneous slices (global and tornado), LightGBM on the large, homogeneous thunderstorm slice. Splitting the database into three regimes was worth it.
- **The GLM is the floor, not the answer.** It stays interpretable and never breaks, but it trails on every slice and goes negative where the data is thinnest.
- **Uncertainty is well calibrated everywhere, while point accuracy varies materially by target and slice.** Conformal coverage is close to 0.90 in all three notebooks; D2 ranges from ~0.67 down to ~0.22 depending on how many casualty events the slice actually contains. Data scarcity, not model choice, is the binding constraint.
- The main lesson is therefore that model complexity should follow data richness: nonlinear/foundation models add value when signal exists, but rare-event uncertainty remains fundamental and should be quantified with adaptive conformal prediction rather than hidden behind point forecasts.
- The results show that very low RMSE or MPD for extremely rare targets can be misleading so D², event counts, actual-versus-predicted plots, and backtesting must be interpreted together.

## Data files

The `data/` folder contains two datasets, each split into Parquet parts (zstd-compressed) to respect GitHub's file-size limits:

- **`StormEvents_part_1..5.parquet`** The raw merged dataset (output of step 2).
- **`StormEvents_fe_ep_augmentation_fin_part_1..10.parquet`** The final dataset with generated narratives, embedding features, and the three LLM-derived columns (output of step 6).

To reassemble a dataset, concatenate its parts in order:

```python
import glob
import pandas as pd

parts = sorted(glob.glob("data/StormEvents_fe_ep_augmentation_fin_part_*.parquet"))
df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
```
