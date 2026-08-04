# Sadanga Gangsa Event Sequence System

Local Streamlit research application for comparing Markov Chain/N-gram, GRU,
and LSTM next-event prediction under a low-resource rhythmic-event sequence
condition. The app models performance-derived tokens; it does not train on raw
audio or claim to generate culturally authentic Sadanga Gangsa music.

## Installation

Python 3.11 through 3.13 is recommended. From this directory, create and
activate a virtual environment, then install the core application dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The core installation supports the Streamlit interface, dataset preparation,
metrics, and Markov Chain/N-gram evaluation. GRU and LSTM are optional because
PyTorch is a substantially larger dependency. To enable the neural models,
install the CPU build after the core requirements:

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cpu -r requirements-neural-cpu.txt
```

The training service explicitly uses the CPU, so a GPU and CUDA are not
required. PyTorch imports are guarded: if PyTorch is unavailable or cannot be
loaded, the app reports that limitation and still permits Markov Chain/N-gram
evaluation.

For development tests:

```powershell
python -m pip install -r requirements-dev.txt
```

The development requirements intentionally do not install PyTorch. Install the
optional CPU neural requirements as shown above when running GRU/LSTM tests.

## Run the app

```powershell
streamlit run app.py
```

## Dataset intake

Use the **Data Intake** page to upload `verified_event_dataset.csv`. The app
requires these columns:

- `group_id`
- `event_index`
- `event_token`

The uploaded file is retained only in the Streamlit session. The app does not
modify, overwrite, or copy the source CSV. Invalid rows are reported and
excluded in memory. Usable rows are sorted by `group_id` and numeric
`event_index`, then converted into one `event_token` sequence per recording.
Unexpected recording IDs or counts produce non-blocking provenance warnings;
the exact upload SHA-256 is retained with saved run metadata.

## Training and evaluation

On the **Protocol** page, select the algorithms and compact model settings.
Window sizes 3, 4, and 5 are supported, with 3 as the default. GRU and LSTM use
small embeddings and hidden states, reproducible seeds, and early stopping to
fit the 586-event dataset on CPU.

Evaluation uses **Leave-One-Recording-Out (LORO)** validation. Each fold holds
out one complete `group_id` for testing and trains on the remaining recordings.
Rows or events are never randomly divided across training and test sets, which
prevents recording-level leakage.

Token IDs are assigned once from the fixed token taxonomy present in the
verified dataset, before LORO folds are created; they are not fitted from token
frequency. Model transition counts and neural parameters are still learned
only from each fold's training recordings. This fixed vocabulary keeps class
IDs comparable across all held-out groups.

Macro F1 deliberately averages over every class in that known vocabulary for
every fold. A class absent from a fold contributes zero rather than being
silently removed from that fold's average, giving all folds the same class
scope.

If PyTorch cannot be loaded, GRU and LSTM failures are shown clearly while
Markov Chain/N-gram evaluation remains available. Metrics and loss values come
only from completed model runs; the application does not create placeholder
results.

## Algorithm code map

The three algorithms have distinct implementation files so their model logic
can be reviewed and explained independently:

- `src/models/markov.py` contains transition counting, additive smoothing,
  unigram fallback, and Markov/N-gram prediction.
- `src/models/gru.py` contains the compact Embedding-GRU-Dropout-Linear
  architecture and its guarded builder.
- `src/models/lstm.py` contains the compact Embedding-LSTM-Dropout-Linear
  architecture and its guarded builder.

`src/services/model_training.py` is the experiment orchestrator: it creates
recording-level LORO jobs, invokes each algorithm, evaluates held-out
predictions, and records real metrics. `src/models/neural_training.py` owns the
shared CPU minibatch loop, temporal validation split, deterministic seeding,
and early stopping used by GRU and LSTM. This separation avoids duplicating
training and metric behavior while preserving a distinct, defense-friendly
model file for each algorithm.

`src/data/result_schema.py` is the single table contract used by the training
orchestrator, artifact validation, and tests.

## Result files

Each saved evaluation receives a unique run directory:

```text
results/
  evaluation/
    runs/
      <run_id>/
        fold_level_results.csv
        algorithm_summary.csv
        training_history.csv       # only when neural epochs exist
        training_config.json       # when a configuration is supplied
        manifest.json
    latest_run.json
```

Required tables are validated before a run is committed. A new run does not
overwrite an earlier experiment, and optional files cannot be carried over
from an older run. The manifest records whether the requested model-fold work
was completed or partial, the requested and completed algorithms, expected and
completed job counts, genuine run errors, and compact dataset provenance
(upload SHA-256, row/group counts, actual group IDs, and the token-ID mapping).
The source CSV itself is never copied into the result folder.
`latest_run.json` points to the most recently saved run and includes its status;
it is not a result table. Files are created only from genuine result rows or
neural training-history rows. Runtime results are excluded from source control
by `.gitignore`.

Fold-level results, per-algorithm summaries, errors, and GRU/LSTM loss
histories are also retained in the active Streamlit session.

## Tests

Most tests are self-contained and use synthetic recording sequences. They do
not need the thesis dataset:

```powershell
python -m pytest -m "not integration"
```

`tests/test_verified_dataset.py` is an integration test for the authoritative
586-event dataset. It expects the existing dataset at this path relative to
`thesis_system`:

```text
../data_pipeline/data/verified_events/verified_event_dataset.csv
```

The test reads that CSV in place; it does not modify or copy it. Run the
dataset integration test, or the complete suite, only when that sibling
project data is available:

```powershell
python -m pytest -m integration
python -m pytest
```

Neural tests are skipped when PyTorch is not installed. The optional-backend
test still verifies that the application imports cleanly and can use Markov
Chain/N-gram without PyTorch.

## Current scope

The working backend covers dataset preparation and comparative
training/evaluation for Markov Chain/N-gram, GRU, and LSTM. The interface also
retains the Generation and Audio stages so the complete thesis workflow stays
visible. Their output actions remain locked unless the required final models,
services, and genuine intermediate records exist; the application never
creates placeholder sequences or audio.
