# AGENTS.md — Thesis System Context for Codex

## 1. Project Identity

This project is a local Streamlit research application for the BSCS thesis:

**Comparative Analysis of Markov Chain, GRU, and LSTM Algorithms for Low-Resource Sadanga Gangsa-Based Rhythmic Event Sequence Generation**

The system is a thesis/research tool. It is not a commercial music generator and it is not meant to claim cultural authenticity.

The main purpose of the app is to support a controlled algorithm comparison among:

1. Markov Chain / N-gram
2. GRU
3. LSTM

The app should focus on **token-based rhythmic-event sequence modeling**, not raw-audio learning.

---

## 2. Main Thesis Framing

Use careful, defense-safe wording throughout the app.

Correct terms:

- Sadanga Gangsa-based rhythmic-event sequences
- performance-derived rhythmic-event dataset
- verified rhythmic-event dataset
- generated rhythmic-event token sequences
- low-resource rhythmic-event sequence modeling
- sample-rendered rhythmic-event simulation

Avoid these claims:

- Do not claim the system generates authentic Sadanga Gangsa music.
- Do not claim the system recreates traditional Sadanga Gangsa performance.
- Do not claim the system represents the full Sadanga/Sinadanga tradition.
- Do not claim the system identifies exact gong identities unless the dataset explicitly supports it.
- Do not describe the rhythmic tokens as N1-N9 gong labels.
- Do not say the models train on raw audio.
- Do not say the isolated strike WAV files are used for algorithm training.

The study compares algorithm behavior under a low-resource rhythmic-event sequence condition.

Defense-safe thesis claim:

> This study compares Markov Chain/N-gram, GRU, and LSTM for low-resource Sadanga Gangsa-based rhythmic-event sequence modeling.

---

## 3. Current App Identity

The Streamlit app is a local Python-based research application.

This file lives at the workspace root and governs the `thesis_system/`
application. Current project structure:

```text
thesis_system/
  app.py
  README.md
  requirements.txt
  requirements-neural-cpu.txt
  requirements-dev.txt
  pytest.ini
  src/
    components/
      ui.py
    data/
      protocol.py
      result_schema.py
      training_config.py
    metrics/
      evaluation.py
      registry.py
    models/
      markov.py
      gru.py
      lstm.py
      pytorch_backend.py
      recurrent.py
      neural_training.py
    pages/
      overview.py
      data_intake.py
      research_protocol.py
      training_workflow.py
      evaluation_methods.py
      generation.py
      audio_rendering.py
      reports.py
    services/
      artifact_store.py
      data_validation.py
      experiment_plan.py
      model_training.py
      sequence_dataset.py
      session_state.py
    styles/
      theme.py
      theme.css
  tests/
```

The app has the main UI pages and a real initial backend for dataset
preparation, LORO training, predictive evaluation, and result export.

The current priority is to keep that backend clear, reproducible, and
defense-safe before implementing final generation.

Do not redesign the whole UI unless necessary.

Preserve the existing minimal style:

grey / black / white theme
clean academic interface
existing page structure
existing component style where possible

## 4. Current Development Status

The app validates and prepares `verified_event_dataset.csv`, builds
recording-level token sequences, creates true LORO folds, and runs real
Markov Chain/N-gram, GRU, and LSTM next-event evaluation on CPU.

The implemented workflow is:

verified_event_dataset.csv
→ validate dataset
→ group token sequences by recording
→ encode event tokens
→ create Leave-One-Recording-Out folds
→ train/evaluate Markov Chain, GRU, and LSTM
→ display fold-level and summary results
→ save versioned evaluation artifacts

The next major implementation stage is final all-recording model training and
16-, 32-, and 64-event token generation.

Sound rendering is not required yet.

Audio rendering should remain future/planned functionality unless specifically requested.

## 5. Main Training Dataset

The correct input file for training/evaluation is:

verified_event_dataset.csv

This file may be uploaded through the Streamlit app.

The minimum required columns for training are:

group_id
event_index
event_token

Useful optional columns:

onset_seconds
ioi_seconds
ioi_category
strength_category
onset_strength_norm
source_id
candidate_event_id
clip_filename
clip_path

The app should validate at least the minimum required columns before allowing training.

## 6. Dataset Description

The dataset contains verified rhythmic events derived from five Sadanga/Sinadanga Gangsa performance recordings.

Expected recording groups:

PERF-001
PERF-002
PERF-003
PERF-004
PERF-005

Expected total event count:

586 rhythmic events

Expected group counts:

PERF-001 = 235 events
PERF-002 = 39 events
PERF-003 = 34 events
PERF-004 = 214 events
PERF-005 = 64 events

The dataset is token-based.

The algorithms train on event_token sequences only.

The algorithms do not train on WAV files.

## 7. Event Token Meaning

The event_token column contains rhythm-based labels created from measurable timing and onset-strength features.

Example tokens:

START_WEAK
START_MEDIUM
START_STRONG
SHORT_WEAK
SHORT_MEDIUM
SHORT_STRONG
MEDIUM_WEAK
MEDIUM_MEDIUM
MEDIUM_STRONG
LONG_WEAK
LONG_MEDIUM
LONG_STRONG

Meaning:

START = first event in a recording
SHORT / MEDIUM / LONG = IOI or timing-gap category
WEAK / MEDIUM / STRONG = onset-strength category

These are rhythmic-event tokens.

They are not exact gong identity labels.

They are not N1-N9 labels.

Do not describe them as exact pitch/gong classes.

## 8. Dataset Provenance and Safe Explanation

The pipeline produced candidate events from ensemble recordings. The candidate clips were reviewed for usable strike sounds. The accepted events were converted into rhythmic-event tokens using measurable timing and onset-strength features.

Safe wording:

The candidate events were reviewed through generated clips for usability. Accepted events were converted into rhythmic-event tokens based on IOI timing and onset-strength features.

Avoid saying:

The model learned exact gong notes.

Avoid saying:

The system identified N1-N9 from ensemble recordings.

Avoid saying:

The output is authentic Sadanga Gangsa music.

## 9. Dataset Loading Rules

When loading verified_event_dataset.csv:

Validate required columns:
group_id
event_index
event_token
Remove invalid rows with missing:
group_id
event_index
event_token
Sort by:
group_id
event_index
Group rows by group_id.
Convert each recording group into a token sequence.

Example:

PERF-001 = START_STRONG → SHORT_MEDIUM → LONG_WEAK → ...
PERF-002 = START_WEAK → MEDIUM_STRONG → SHORT_WEAK → ...
Encode tokens into integer IDs.
Preserve:
token-to-ID mapping
ID-to-token mapping
## 10. Validation Method

Use Leave-One-Recording-Out validation.

Since there are five recording groups, create five folds:

Fold 1: test PERF-001, train on PERF-002, PERF-003, PERF-004, PERF-005
Fold 2: test PERF-002, train on PERF-001, PERF-003, PERF-004, PERF-005
Fold 3: test PERF-003, train on PERF-001, PERF-002, PERF-004, PERF-005
Fold 4: test PERF-004, train on PERF-001, PERF-002, PERF-003, PERF-005
Fold 5: test PERF-005, train on PERF-001, PERF-002, PERF-003, PERF-004

Do not randomly split individual rows/events across training and testing.

Random row-level splitting would cause leakage because events from the same recording could appear in both train and test sets.

Defense-safe explanation:

Each recording was treated as a separate group. The models were trained on four recordings and tested on one unseen recording using leave-one-recording-out validation.

## 11. Training Task

The task is next-event prediction.

Given a short window of previous rhythmic-event tokens, predict the next token.

Example:

Input:
SHORT_WEAK → MEDIUM_STRONG → SHORT_MEDIUM

Target:
LONG_WEAK

Use small windows because the dataset is small.

Recommended window sizes:

3
4
5

Default recommended window size:

3
## 12. Window Creation

For each recording sequence, create sliding windows.

Example sequence:

START_STRONG → SHORT_MEDIUM → MEDIUM_WEAK → LONG_STRONG → SHORT_WEAK

With window size 3:

Input:  START_STRONG → SHORT_MEDIUM → MEDIUM_WEAK
Target: LONG_STRONG

Input:  SHORT_MEDIUM → MEDIUM_WEAK → LONG_STRONG
Target: SHORT_WEAK

Do not create windows across different recordings.

Each recording must remain a separate sequence.

## 13. Algorithm 1 — Markov Chain / N-gram

The Markov Chain / N-gram model is the simple interpretable baseline.

It should learn transition probabilities from training sequences.

Example:

Given previous token(s), what token usually comes next?

Implementation notes:

Support N-gram order 1 or 2.
Include unigram fallback.
Use smoothing to avoid zero-probability issues.
Evaluate next-token predictions on the held-out recording.
Markov Chain does not use epochs.
Markov Chain does not have neural training loss curves.
Report training time and prediction metrics.

Good UI explanation:

The Markov Chain/N-gram baseline learns local transition probabilities between rhythmic-event tokens.

## 14. Algorithm 2 — GRU

The GRU model should be a small recurrent neural model for next-token prediction.

Because the dataset is small, keep the model compact.

Recommended settings:

embedding_dim = 8 or 16
hidden_units = 16 or 32
dropout = 0.2 to 0.4
batch_size = 8 or 16
epochs = 30 to 100
early_stopping = enabled
window_size = 3 to 5

Track:

training_loss
validation_loss
training_accuracy if available
validation_accuracy if available
training_time_seconds
epochs_completed

Good UI explanation:

The GRU model learns token-sequence patterns using a compact recurrent memory structure.

## 15. Algorithm 3 — LSTM

The LSTM model should also be small.

Recommended settings:

embedding_dim = 8 or 16
hidden_units = 16 or 32
dropout = 0.2 to 0.4
batch_size = 8 or 16
epochs = 30 to 100
early_stopping = enabled
window_size = 3 to 5

LSTM may overfit because the dataset is small.

The app should clearly show training/validation behavior.

Good UI explanation:

The LSTM model uses a recurrent memory mechanism that may capture longer token patterns, but it can overfit under a small-data condition.

## 16. Dependency Handling

Do not make the whole app fail if PyTorch or TensorFlow is missing.

Markov Chain should work with:

standard Python
pandas
numpy

For GRU and LSTM:

Use available installed libraries.
Prefer PyTorch if already available.
If no neural-network library is available, show a clear Streamlit warning:
GRU/LSTM training requires PyTorch or TensorFlow.
Markov Chain training can still run.

Do not require GPU.

The app should run on CPU.

## 17. Metrics

For each algorithm and each fold, compute:

accuracy
macro_f1
top_k_accuracy
cross_entropy_loss or negative_log_loss if available
training_time_seconds

For GRU and LSTM, also store:

epochs_completed
final_training_loss
final_validation_loss

A fold-level result table should include:

algorithm
fold
test_group
train_groups
train_event_count
test_event_count
window_size
vocabulary_size
accuracy
macro_f1
top_k_accuracy
loss
training_time_seconds
epochs_completed
final_training_loss
final_validation_loss

Not every metric will apply to every algorithm.

For Markov Chain, neural-specific fields may be blank or N/A.

## 18. Overfitting Checks

For GRU and LSTM, compare training and validation/test behavior.

Signs of overfitting:

training loss decreases
validation loss increases
training accuracy is high
test accuracy is much lower

The app should not hide poor results.

Poor results are still useful because the thesis is comparative.

Defense-safe explanation:

Because the dataset is small, overfitting behavior is part of the comparison. The study evaluates whether the recurrent models improve prediction or simply memorize the limited training recordings.

## 19. Generation

After formal evaluation, final models may be trained on all five recordings.

Final models are used for generation, not fold-level testing.

Generation lengths:

16 events
32 events
64 events

Each algorithm should generate token sequences such as:

START_STRONG → SHORT_MEDIUM → SHORT_STRONG → LONG_WEAK → ...

Generation is token-based.

Generated sequences do not have one correct answer, so do not evaluate them mainly using accuracy.

Generation behavior metrics may include:

token_diversity
unique_token_count
repetition_rate
transition_validity
distribution_similarity
generation_stability

Good UI explanation:

After evaluation, final models may be trained on all verified recordings to generate Sadanga Gangsa-based rhythmic-event token sequences of selected lengths.

## 20. Sound Rendering Status

Sound rendering is not required in the current implementation.

Current priority:

training/evaluation dataset
→ algorithm comparison
→ generated token sequences

Later priority:

selected strike clips
→ performance-derived sound bank
→ rendered audio simulation

Do not connect full audio rendering unless specifically requested.

## 21. Sound Rendering Future Plan

Later, sound rendering should use a separate performance-derived sample bank.

The old isolated N1-N9 strike bank should not be used as the main rendering bank for the current rhythm-token method, because the current tokens are not N1-N9 labels.

Future rendering plan:

Generated token: SHORT_STRONG
SHORT = timing gap
STRONG = choose a strong strike sample

Possible future sample bank structure:

data/sample_bank/performance_derived/
  WEAK/
  MEDIUM/
  STRONG/
  sample_bank_metadata.csv

Use one shared sample bank for all algorithms.

Do not create a different sound bank per algorithm.

## 22. Important Separation

Training/evaluation data:

verified_event_dataset.csv

Sound simulation data:

candidate_event_clips or selected strike WAV samples

The algorithms train on token sequences only.

They do not train on WAV files.

## 23. Data Pipeline Context

The dataset was created through a separate data pipeline.

Relevant pipeline stages:

01_build_inventory.py
02_prepare_curation_review.py
03_screen_isolated_audio.py
04_prepare_ensemble_audio.py
05_detect_ensemble_events.py
06_prepare_ensemble_event_review.py
07_build_verified_event_dataset.py

Step 07 outputs:

data/event_review/ensemble_event_review_completed.csv
data/verified_events/verified_event_dataset.csv
data/verified_events/tokenization_summary.csv

For the Streamlit app, the main input is:

verified_event_dataset.csv

The app does not need to rerun the data pipeline unless specifically requested.

## 24. Expected App Behavior

The Streamlit app should allow the user to:

Upload verified_event_dataset.csv.
Validate required columns.
Show dataset summary.
Show token distribution.
Show recording group counts.
Configure algorithm options.
Configure window size.
Run Leave-One-Recording-Out evaluation.
Train/evaluate Markov Chain, GRU, and LSTM.
Show fold-level results.
Show average results per algorithm.
Show GRU/LSTM loss curves if implemented.
Train final models on all recordings after evaluation.
Generate 16-, 32-, and 64-event token sequences.
Export result tables and generated sequences as CSV.
## 25. Page Responsibilities
data_intake.py

Should handle:

dataset upload
required-column validation
dataset preview
dataset summary
group count summary
token distribution summary
training_workflow.py

Should handle:

algorithm selection
training configuration
window size
Leave-One-Recording-Out execution
fold-level result display
training status
evaluation_methods.py

Should handle:

metric explanations
results summary
algorithm comparison tables
optional plots
generation.py

Should handle:

final model training status
sequence length selection
temperature/top-k/top-p if implemented
generated token sequence display
export generated sequences
audio_rendering.py

Keep as planned/future functionality unless specifically requested.

reports.py

Should eventually show/export:

dataset summary
training results
evaluation tables
generation outputs
## 26. Recommended Backend Modules

Prefer adding backend service modules instead of placing all logic inside page files.

Possible service files:

src/services/sequence_dataset.py
src/services/model_training.py
src/services/generation_service.py

Possible metric files:

src/metrics/evaluation.py
src/metrics/generation_metrics.py

Suggested responsibilities:

sequence_dataset.py
= loading, validation, grouping, encoding, folds, windows

model_training.py
= Markov, GRU, LSTM training/evaluation

generation_service.py
= final model training and token sequence generation

evaluation.py
= accuracy, macro F1, top-k accuracy, loss

generation_metrics.py
= diversity, repetition, transition validity
## 27. Recommended Implementation Order

Implement in this order:

Dataset loading and validation.
Dataset summary.
Grouping by group_id.
Token encoding.
Leave-One-Recording-Out fold creation.
Window creation for next-event prediction.
Markov Chain training/evaluation.
GRU training/evaluation.
LSTM training/evaluation.
Fold-level result table.
Summary result table.
Final model training.
Token sequence generation.
Export outputs.

Do not implement audio rendering yet unless explicitly requested.

## 28. Expected Training Output Files

Each evaluation with at least one genuine fold result is saved in its own run
folder:

```text
results/
  evaluation/
    runs/
      <run_id>/
        fold_level_results.csv
        algorithm_summary.csv
        training_history.csv
        training_config.json
        manifest.json
    latest_run.json
```

Optional files are present only when the run produced their real content.
The manifest records complete/partial status, requested versus completed jobs,
errors, and compact dataset provenance without copying the source CSV. Final
generation-model artifacts will use separate generation/model locations when
that stage is implemented.

Create required folders safely and do not crash if they do not exist.

## 29. Coding Rules for Codex

Before editing files:

Read this AGENTS.md.
Inspect the existing project structure.
Summarize which files will be modified or created.
Do not edit until the user approves, unless the user already explicitly asked for direct editing.
Preserve the existing UI style.
Avoid unnecessary redesign.
Avoid removing existing helper functions unless they are truly unused.
Do not modify __pycache__ files.
Do not delete current pages.
Prefer backend service modules for training logic.
Keep page files focused on UI.
Keep code readable and defense-friendly.
Add comments where thesis logic needs explanation.
Keep the app runnable even if GRU/LSTM dependencies are missing.
Never silently fake metrics or model results.
## 30. Result Integrity Rules

Do not fabricate training results.

Do not hard-code fake accuracy, loss, F1-score, or generation results.

If a model cannot train, show a clear message.

If a metric cannot be computed, show N/A or an explanatory warning.

If the dataset is too small for a chosen setting, warn the user and suggest smaller settings.

The app should be honest about limitations.

## 31. Defense-Safe UI Text

Use this explanation for dataset:

The uploaded dataset contains verified performance-derived rhythmic events converted into token sequences. Each recording is treated as a separate group for leave-one-recording-out validation.

Use this explanation for training:

The models are trained for next-event prediction. Given a short sequence of previous rhythmic-event tokens, each model predicts the next token.

Use this explanation for validation:

Leave-one-recording-out validation holds out one complete performance recording for testing while training on the remaining recordings. This reduces leakage between training and testing data.

Use this explanation for generation:

After evaluation, final models may be trained on all verified recordings to generate Sadanga Gangsa-based rhythmic-event token sequences of selected lengths.

Use this explanation for limitation:

Because the dataset is small and focused on five recordings, results should be interpreted as comparative insights under a low-resource condition rather than broad generalization to all Sadanga Gangsa performance.

## 32. Important Non-Goals

Do not implement these unless specifically requested:

full audio rendering
automatic cultural authenticity scoring
N1-N9 gong identity classification
raw audio model training
Transformer model
database integration
major UI redesign

Current non-goal:

Do not implement full audio rendering yet.
## 33. Current Priority

The implemented priority is:

Keep the Streamlit app able to train and evaluate Markov Chain, GRU, and LSTM
using `verified_event_dataset.csv` with reproducible, recording-level
validation and genuine versioned results.

The current app workflow is:

Upload verified_event_dataset.csv
→ validate
→ show dataset summary
→ configure training
→ run LORO evaluation
→ display fold-level results
→ display summary comparison

Final all-recording training and token generation remain the next planned
implementation stage. Full audio rendering remains future work.
## 34. Final Reminder

The app should support this thesis statement:

This study compares Markov Chain/N-gram, GRU, and LSTM for low-resource Sadanga Gangsa-based rhythmic-event sequence modeling.

The app should not claim:

This system generates authentic Sadanga Gangsa music.
