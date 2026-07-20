# ### Import the libraries

import pandas as pd
import numpy as np
import os
import json
import time
from pathlib import Path

# ### Connect to the OpenAI API
# Requires `OPENAI_API_KEY` in the `.env` file in this folder 

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  
client = OpenAI()  

# ### Load the cleaned dataset
# 

df = pd.read_csv('./work/StormEvents_cleaned.csv', low_memory=False)
print(f"Loaded df: {df.shape}")

# ### Identify rows needing narrative generation
# Here we target the exact set that needs an API call: any row missing `EPISODE_NARRATIVE` **or** `EVENT_NARRATIVE`, wherever it falls in the dataset. Fully-complete rows are skipped entirely so we never spend tokens on rows that don't need changes.

episode_empty = df["EPISODE_NARRATIVE"].isna()
event_empty = df["EVENT_NARRATIVE"].isna()
both_empty = episode_empty & event_empty
either_empty = episode_empty | event_empty

need_idx = df.index[either_empty]

print(f"Total rows: {len(df):,}")
print(f"Both narratives empty: {both_empty.sum():,}")
print(f"Mixed (one present, one missing): {(either_empty & ~both_empty).sum():,}")
print(f"Fully complete (skipped): {(~either_empty).sum():,}")
print(f"Rows to submit to the API: {len(need_idx):,}")

# ### Configuration for the narrative-generation job
# For mixed rows, the existing narrative is passed to the model as context so the generated one stays consistent; the write-back step later only overwrites the cell(s) that were actually blank.

RUN_NARRATIVE_API = True                
SAMPLE_N          = None                 
MODEL             = "gpt-4o-mini"        
BATCH_DIR         = Path("./work/batches")
OUT_PATH          = "./work/filled/StormEvents_narratives_filled_fin.csv"
MAX_REQUESTS_PER_BATCH = 45_000          

CONTEXT_COLS = [
    "EVENT_TYPE", "EVENT_GROUP", "STATE", "CZ_NAME", "BEGIN_DATE_TIME", "END_DATE_TIME",
    "BEGIN_LOCATION", "END_LOCATION", "MAGNITUDE", "MAGNITUDE_TYPE", "TOR_F_SCALE",
    "TOR_LENGTH", "TOR_WIDTH", "FLOOD_CAUSE", "DAMAGE_PROPERTY", "DAMAGE_CROPS",
    "INJURIES_DIRECT", "INJURIES_INDIRECT", "DEATHS_DIRECT", "DEATHS_INDIRECT",
    "SOURCE", "WFO",
]

SYSTEM_PROMPT = (
    "You write factual, concise NOAA Storm Events narratives from structured fields. "
    "Return ONLY a JSON object with two keys:\n"
    "  episode_narrative: 1 sentence on the broader weather episode/setup.\n"
    "  event_narrative:   1 sentence on this specific event and its impacts.\n"
    "If an existing narrative is supplied for one of the two fields, reuse it verbatim "
    "for that key and only compose the missing one, keeping the two consistent. "
    "Use only the facts provided; do not invent places, names, or numbers that are not given. "
    "Plain past-tense prose, no preamble."
)

NARRATIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "episode_narrative": {"type": "string"},
        "event_narrative": {"type": "string"},
    },
    "required": ["episode_narrative", "event_narrative"],
    "additionalProperties": False,
}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "narrative", "schema": NARRATIVE_SCHEMA, "strict": True},
}


def _row_to_prompt(row):
    facts = "; ".join(f"{c}={row[c]}" for c in CONTEXT_COLS if c in row and pd.notna(row[c]))
    parts = [f"Storm event fields: {facts}."]
    if pd.notna(row["EPISODE_NARRATIVE"]):
        parts.append(f"Existing episode_narrative: {row['EPISODE_NARRATIVE']}")
    if pd.notna(row["EVENT_NARRATIVE"]):
        parts.append(f"Existing event_narrative: {row['EVENT_NARRATIVE']}")
    return " ".join(parts)

# ### Build JSONL batch request files
# OpenAI's Batch API takes a JSONL file of request objects. Each line's `custom_id` carries the **original dataframe index**. That's the key that lets every generated narrative be written back into its own row later. 

# %%
def _make_request(idx, prompt):
    return {
        "custom_id": f"row-{idx}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "max_tokens": 500,
            "response_format": RESPONSE_FORMAT,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        },
    }


todo = need_idx if SAMPLE_N is None else need_idx[:SAMPLE_N]
print(f"Rows selected for this run: {len(todo):,}")

batch_files = []
if RUN_NARRATIVE_API:
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    prompts = df.loc[todo].apply(_row_to_prompt, axis=1)

    for start in range(0, len(todo), MAX_REQUESTS_PER_BATCH):
        chunk_idx = todo[start:start + MAX_REQUESTS_PER_BATCH]
        path = BATCH_DIR / f"batch_input_{start}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for idx in chunk_idx:
                f.write(json.dumps(_make_request(idx, prompts[idx])) + "\n")
        batch_files.append(path)
        print(f"  wrote {path} ({len(chunk_idx):,} requests)")
else:
    print("RUN_NARRATIVE_API is False - skipping batch file creation.")

# ### Submit batches to OpenAI
# The job manifest (batch id + source file) is written to disk so polling can resume after a kernel restart without resubmitting anything.

batch_jobs = []  
JOBS_PATH = BATCH_DIR / "batch_jobs.json"

if RUN_NARRATIVE_API:
    for path in batch_files:
        uploaded = client.files.create(file=open(path, "rb"), purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        batch_jobs.append({"batch_id": batch.id, "input_file": str(path)})
        print(f"  submitted {batch.id}  <- {path}")

    with open(JOBS_PATH, "w", encoding="utf-8") as f:
        json.dump(batch_jobs, f, indent=2)
    print(f"Saved job manifest -> {JOBS_PATH}")
else:
    print("RUN_NARRATIVE_API is False - skipping submission.")

# ### Check batch status
# The batches run on OpenAI's servers. This cell just checks current status once and reports it.

WAIT_FOR_COMPLETION = False  
TERMINAL = {"completed", "failed", "expired", "cancelled"}

if RUN_NARRATIVE_API:
    with open(JOBS_PATH, encoding="utf-8") as f:
        batch_jobs = json.load(f)

    def _check_once():
        statuses = {}
        for job in batch_jobs:
            b = client.batches.retrieve(job["batch_id"])
            statuses[job["batch_id"]] = b.status
            done = getattr(b.request_counts, "completed", "?")
            total = getattr(b.request_counts, "total", "?")
            print(f"  {job['batch_id']} -> {b.status}  ({done}/{total} requests)")
        return statuses

    if WAIT_FOR_COMPLETION:
        pending = {job["batch_id"] for job in batch_jobs}
        while pending:
            statuses = _check_once()
            pending = {bid for bid, s in statuses.items() if s not in TERMINAL}
            if pending:
                print(f"  still processing: {len(pending)} - sleeping 30s")
                time.sleep(30)
        print("All batches finished.")
    else:
        statuses = _check_once()
        if all(s in TERMINAL for s in statuses.values()):
            print("All batches finished - safe to run the next cell.")
        else:
            print("Still processing - re-run this cell later.")
else:
    print("RUN_NARRATIVE_API is False - skipping status check.")

# ### Parse results and merge back into the full dataset
# This is the answer to "how do I rebuild the whole dataset in the same order": `df_all` starts as a full copy of `df`, keeping its original row order and index. Results are never concatenated, each one is written into its own row via `df_all.at[idx, ...]`, where `idx` is parsed back out of `custom_id`. So the output order matches the input order by construction, no matter what order the Batch API returns results in.

df_all = df.copy()
df_all["NARRATIVE_SOURCE"] = np.where(either_empty, "pending", "original")

failed_ids = []

if RUN_NARRATIVE_API:
    with open(JOBS_PATH, encoding="utf-8") as f:
        batch_jobs = json.load(f)

    # Guard against "Run All": fail loudly instead of silently saving an
    # incomplete file if a batch hasn't finished processing yet.
    IN_PROGRESS = {"validating", "in_progress", "finalizing", "cancelling"}
    live_statuses = {job["batch_id"]: client.batches.retrieve(job["batch_id"]).status for job in batch_jobs}
    still_running = [bid for bid, s in live_statuses.items() if s in IN_PROGRESS]
    if still_running:
        raise RuntimeError(
            f"{len(still_running)} batch(es) still running: {still_running}. "
            "Re-run the status-check cell later (or set WAIT_FOR_COMPLETION=True there) "
            "before running this cell."
        )
    not_completed = {bid: s for bid, s in live_statuses.items() if s != "completed"}
    if not_completed:
        print(f"Warning: batch(es) ended without 'completed' status: {not_completed} - their rows will stay blank.")

    filled = 0
    for job in batch_jobs:
        batch = client.batches.retrieve(job["batch_id"])

        if batch.error_file_id:
            err_content = client.files.content(batch.error_file_id).text
            for line in err_content.splitlines():
                rec = json.loads(line)
                failed_ids.append(rec["custom_id"])

        if not batch.output_file_id:
            continue

        out_content = client.files.content(batch.output_file_id).text
        for line in out_content.splitlines():
            rec = json.loads(line)
            idx = int(rec["custom_id"].split("-", 1)[1])

            if rec.get("error") or rec["response"]["status_code"] != 200:
                failed_ids.append(rec["custom_id"])
                continue

            message = rec["response"]["body"]["choices"][0]["message"]["content"]
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                failed_ids.append(rec["custom_id"])
                continue

            if pd.isna(df_all.at[idx, "EPISODE_NARRATIVE"]):
                df_all.at[idx, "EPISODE_NARRATIVE"] = data.get("episode_narrative", "")
            if pd.isna(df_all.at[idx, "EVENT_NARRATIVE"]):
                df_all.at[idx, "EVENT_NARRATIVE"] = data.get("event_narrative", "")
            df_all.at[idx, "NARRATIVE_SOURCE"] = "llm_generated"
            filled += 1

    print(f"Filled {filled:,} rows from the API.")
    if failed_ids:
        print(f"Failed/unparsed requests: {len(failed_ids):,} (see failed_ids)")
else:
    print("RUN_NARRATIVE_API is False - df_all is a copy of df with narratives still blank where missing.")

# ### Save the filled dataset + sanity checks
# Row count and `EVENT_ID` order are checked against the original `df` before saving, to confirm nothing got reordered or dropped.

assert len(df_all) == len(df), "row count changed"
assert (df_all["EVENT_ID"].values == df["EVENT_ID"].values).all(), "row order changed"

df_all.to_csv(OUT_PATH, index=False)
print(f"Saved df_all -> {OUT_PATH}  shape: {df_all.shape}")
print(f"Remaining empty EPISODE_NARRATIVE: {df_all['EPISODE_NARRATIVE'].isna().sum():,}")
print(f"Remaining empty EVENT_NARRATIVE:   {df_all['EVENT_NARRATIVE'].isna().sum():,}")




