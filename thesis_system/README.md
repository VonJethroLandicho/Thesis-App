# Sadanga Gangsa System

Local Streamlit research application for comparing Markov Chain/N-gram, GRU,
and LSTM under a low-resource rhythmic-event sequence condition. The app uses
performance-derived rhythmic-event tokens. It does not train the algorithms on
raw WAV audio and does not claim that generated output is an authentic Sadanga
Gangsa performance.

## Install

Python 3.11 through 3.13 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To enable GRU and LSTM on CPU:

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cpu -r requirements-neural-cpu.txt
```

For tests:

```powershell
python -m pip install -r requirements-dev.txt
```

## Run

```powershell
streamlit run app.py
```

## User workflow

The interface is organized as two guided workflows instead of independent menu
pages.

### Compare Algorithms — complete this first

1. **Prepare Data** — upload and validate `verified_event_dataset.csv`.
2. **Choose Settings** — keep one reproducible test configuration for the
   algorithms being compared.
3. **Train & Test** — run recording-level Leave-One-Recording-Out evaluation.
4. **Compare Results** — review visual comparisons, held-out recording behavior,
   learning curves, and detailed result tables.
5. **Save Results** — download genuine dataset, configuration, metric, history,
   and error records from the session.

### Generate & Listen — unlocked after a complete comparison

1. **Choose Algorithm** — select an evaluated algorithm using the comparison as
   evidence.
2. **Train Final Model** — train one final model on all verified recordings for
   generation only.
3. **Generate Sequence** — create a bounded rhythmic-event token sequence.
4. **Prepare Samples** — validate the shared performance-derived WAV sample
   bank used only for rendering.
5. **Create & Listen** — render the generated tokens as a sample-based mono WAV
   research preview.
6. **Save Output** — download the generated sequence, preview, mapping log, and
   summary.

Every workflow page includes a sticky step navigator. Locked Next actions explain
what prerequisite is missing both visibly and through hover help. Contextual
"What happens when I do this?" cues explain important primary actions without
making those explanations mandatory.

## Dataset

The training/evaluation input is `verified_event_dataset.csv` with at least:

- `group_id`
- `event_index`
- `event_token`

The file is read into the Streamlit session and is not overwritten by the app.
Usable rows are ordered within each recording and converted into one token
sequence per recording.

## Evaluation

The study uses **Leave-One-Recording-Out (LORO)** validation. One complete
recording is held out for testing while the remaining recordings are used for
training. Individual rows are not randomly mixed between training and testing.

Current comparison records include accuracy, Macro F1, top-k accuracy,
prediction loss, training time, and neural training history where applicable.
The Results screen presents these first as plain-language visual comparisons,
then exposes fold-level and technical records under detailed sections.

The application never inserts fake scores when a run fails. Failed or
unavailable results remain missing and are recorded as errors.

## Code layout

```text
thesis_system/
  app.py
  .streamlit/
    config.toml
  src/
    components/       # reusable UI and navigation
    content/          # plain-language UI text and glossary
    screens/
      compare/        # 5-step comparison workflow
      generate/       # 6-step generation/audio workflow
    workflows/        # routes, guards, step availability/progress
    data/             # protocol and result data structures
    metrics/          # evaluation calculations and metric registry
    models/           # Markov, GRU, LSTM implementations
    services/         # validation, training, generation, audio, artifacts
    styles/           # Streamlit theme loader and CSS
  tests/
```

`app.py` uses `st.Page` and `st.navigation(position="hidden")` for routing while
the application renders its own workflow UI.

## Result artifacts

Evaluation runs with genuine results are stored in versioned run folders:

```text
results/
  evaluation/
    runs/
      <run_id>/
        fold_level_results.csv
        algorithm_summary.csv
        training_history.csv       # when neural history exists
        training_config.json
        manifest.json
    latest_run.json
```

Generated/model and audio outputs remain separate from the held-out evaluation
records.

## Tests

Most tests use synthetic data:

```powershell
python -m pytest -m "not integration"
```

The authoritative dataset integration test expects the sibling project path:

```text
../data_pipeline/data/verified_events/verified_event_dataset.csv
```

Run the complete suite when that sibling data is present:

```powershell
python -m pytest
```

PyTorch-specific tests are skipped when the optional neural dependency is not
installed.
