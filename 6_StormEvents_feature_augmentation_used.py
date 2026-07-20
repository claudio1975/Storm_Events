# # Features Data Augmentation by LLM to Every Storm Event
# 
# | New column | Question the LLM answers | Possible answers |
# |---|---|---|
# | `risk` | How dangerous was the episode? | high / medium / low |
# | `impact_type` | What was mainly affected? | casualties / property_damage / crop_damage / infrastructure_disruption / no_significant_impact |
# | `event_scope` | How large an area was affected? | localized / county-wide / regional / widespread |
# 
# All three are read from **`EPISODE_NARRATIVE`**. 

# ---
# ## Step 1 — Load the tools
# 

import pandas as pd
import numpy as np
import os
import json
import time
from pathlib import Path

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# ## Step 2 — Connect to the OpenAI service
# 

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()          
client = OpenAI()      
print("Connected: OpenAI key loaded." if os.getenv("OPENAI_API_KEY") else "PROBLEM: no key found!")

# ## Step 3 — Load the storm events data
# 

df = pd.read_csv('./work/StormEvents_fe_embedding_fin.csv', low_memory=False)
print(f"Loaded {len(df):,} storm events with {df.shape[1]} columns.")


# ## Step 4 — Settings
# 
# Everything adjustable lives in this one cell:
# 
# - **`RUN_FEATURE_API`** — the master switch for the *run* steps (building and submitting
#   the batches). While `False`, nothing is sent to OpenAI.
#   
# - **`SAMPLE_N`** — the safety valve for testing. Set it to a small number (e.g. `100`) to
#   submit only the first 100 events and check the quality of the answers before 
#   the full run. Set it to `None` for the real, complete run.
#   
# - **`SYSTEM_PROMPT`** — the instructions given to the AI, with a plain definition of every
#   answer so the labels are consistent.
#   

RUN_FEATURE_API = True              
SAMPLE_N        = None              
MODEL           = "gpt-4o-mini"     

BATCH_DIR    = Path("./work/variable_batches")    
JOBS_PATH    = BATCH_DIR / "batch_jobs.json"        
OUT_PATH     = "./work/StormEvents_fe_ep_augmentation_fin.csv"  
MAX_REQUESTS_PER_BATCH = 45_000

RISK_LABELS   = ["high", "medium", "low"]
IMPACT_LABELS = ["casualties", "property_damage", "crop_damage",
                 "infrastructure_disruption", "no_significant_impact"]
SCOPE_LABELS  = ["localized", "county-wide", "regional", "widespread"]

SYSTEM_PROMPT = (
    "You classify NOAA storm episode descriptions. Read the description and answer three "
    "questions about it. Base every answer only on what the text says; do not assume "
    "impacts that are not mentioned.\n"
    "1. risk - overall danger the episode posed to people and property:\n"
    "   high   = deaths, injuries, or major/widespread destruction occurred or were clearly likely.\n"
    "   medium = real but limited damage or disruption (downed trees, localized flooding, "
    "power outages, minor property damage).\n"
    "   low    = little or no impact on people or property.\n"
    "2. impact_type - the main impact the episode had:\n"
    "   casualties = people were killed or injured.\n"
    "   property_damage = homes, businesses, vehicles, boats or other private property "
    "were damaged (including trees falling on them).\n"
    "   crop_damage = crops, orchards, timber, livestock or farmland were damaged.\n"
    "   infrastructure_disruption = downed power lines, power outages, blocked or "
    "washed-out roads, damaged bridges, or disrupted public utilities, transport or "
    "communications.\n"
    "   no_significant_impact = the description mentions no specific impact.\n"
    "   If more than one impact applies, choose the first that applies in this order: "
    "casualties, property_damage, crop_damage, infrastructure_disruption.\n"
    "3. event_scope - how large an area the episode affected:\n"
    "   localized   = one spot, town or small part of a county.\n"
    "   county-wide = most of a single county.\n"
    "   regional    = several counties or a large part of a state.\n"
    "   widespread  = multiple states or an entire region.\n"
    "Rules: trees down in open areas with no other damage mentioned means "
    "no_significant_impact. Damage must be stated in the text, not inferred from the "
    "event type."
)

FEATURE_SCHEMA = {
    "type": "object",
    "properties": {
        "risk":        {"type": "string", "enum": RISK_LABELS},
        "impact_type": {"type": "string", "enum": IMPACT_LABELS},
        "event_scope": {"type": "string", "enum": SCOPE_LABELS},
    },
    "required": ["risk", "impact_type", "event_scope"],
    "additionalProperties": False,
}
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "storm_features", "schema": FEATURE_SCHEMA, "strict": True},
}


# ## Step 5 — Build the batch request files
# 

def _make_request(idx, text):
    """One question for the Batch service: three answers about one event's episode description."""
    return {
        "custom_id": f"row-{idx}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "max_tokens": 60,                       
            "response_format": RESPONSE_FORMAT,     
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        },
    }


# Every event that has an episode description.
need_idx = df.index[df["EPISODE_NARRATIVE"].notna()]
todo = need_idx if SAMPLE_N is None else need_idx[:SAMPLE_N]
mode = "FULL run" if SAMPLE_N is None else f"TEST run (first {SAMPLE_N} events)"
print(f"This is a {mode}: {len(todo):,} of {len(df):,} events will be classified.")

batch_files = []
if RUN_FEATURE_API:
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(todo), MAX_REQUESTS_PER_BATCH):
        chunk_idx = todo[start:start + MAX_REQUESTS_PER_BATCH]
        path = BATCH_DIR / f"batch_input_{start}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for idx in chunk_idx:
                f.write(json.dumps(_make_request(idx, df.at[idx, "EPISODE_NARRATIVE"])) + "\n")
        batch_files.append(path)
        print(f"  wrote {path} ({len(chunk_idx):,} requests)")
else:
    print("RUN_FEATURE_API is False - skipping batch file creation.")


# ## Step 6 — Submit the batches to OpenAI
# 

batch_jobs = []

if RUN_FEATURE_API:
    if JOBS_PATH.exists():
        raise RuntimeError(
            f"{JOBS_PATH} already exists - batches were already submitted. "
                    )

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
    print("RUN_FEATURE_API is False - skipping submission.")


# ## Step 7 — Check whether the batches are done
# 

WAIT_FOR_COMPLETION = False
TERMINAL = {"completed", "failed", "expired", "cancelled"}

if JOBS_PATH.exists():
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
    print(f"No job manifest at {JOBS_PATH} - submit the batches first (Steps 5-6).")


# ## Step 8 — Collect the answers and add the three new columns

if not JOBS_PATH.exists():
    raise RuntimeError(
        f"No job manifest at {JOBS_PATH} - nothing to collect. "
        "Run Steps 5-6 first to submit the batches."
    )

df_all = df.copy()
# Start with empty typed columns, so the answers can be written into them.
for col in ["risk", "impact_type", "event_scope"]:
    df_all[col] = pd.Series(pd.NA, index=df_all.index, dtype="string")
failed_ids = []

with open(JOBS_PATH, encoding="utf-8") as f:
    batch_jobs = json.load(f)

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

rated = 0
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
            answers = json.loads(message)
        except json.JSONDecodeError:
            failed_ids.append(rec["custom_id"])
            continue

        # Accept the answer only if every field is present and valid.
        valid = (
            answers.get("risk") in RISK_LABELS
            and answers.get("impact_type") in IMPACT_LABELS
            and answers.get("event_scope") in SCOPE_LABELS
        )
        if not valid:
            failed_ids.append(rec["custom_id"])
            continue

        df_all.at[idx, "risk"] = answers["risk"]
        df_all.at[idx, "impact_type"] = answers["impact_type"]
        df_all.at[idx, "event_scope"] = answers["event_scope"]
        rated += 1

print(f"Events classified: {rated:,} of {len(df_all):,} ({df_all['risk'].notna().mean():.1%})")
if failed_ids:
    print(f"Failed/unreadable answers: {len(failed_ids):,} (see failed_ids)")


# ## Step 9 — Review the result (and save, on a full run)

labeled = df_all[df_all["risk"].notna()]

if len(labeled):
    for col in ["risk", "impact_type", "event_scope"]:
        print(f"How the classified events split across {col}:")
        print(labeled[col].value_counts().to_string(), "\n")

    preview_cols = ["EVENT_TYPE", "STATE", "EPISODE_NARRATIVE",
                    "risk", "impact_type", "event_scope"]
    preview = labeled[preview_cols].assign(
        EPISODE_NARRATIVE=labeled["EPISODE_NARRATIVE"].str.slice(0, 120) + "...")
    display(preview.head(10).style.hide(axis="index"))

# Sanity checks: nothing lost, nothing shuffled, nothing empty.
assert len(df_all) == len(df), "row count changed"
assert (df_all["EVENT_ID"].values == df["EVENT_ID"].values).all(), "row order changed"

if SAMPLE_N is not None:
    print(f"\nTest run: {len(labeled):,} events classified. "
          "Nothing saved yet - set SAMPLE_N = None in Step 4 for the full run.")
else:
    # Fail loudly rather than save a dataset with empty feature columns.
    coverage = df_all["risk"].notna().mean()
    if coverage == 0:
        raise RuntimeError("No answers were collected - refusing to save. "
                           "Check Steps 7-8 before saving.")
    if coverage < 0.95:
        print(f"Warning: only {coverage:.1%} of events were classified - "
              "saving anyway, but check the failed batches.")
    df_all.to_csv(OUT_PATH, index=False)
    print(f"\nFull enriched dataset saved: {len(df_all):,} events -> {OUT_PATH}")



