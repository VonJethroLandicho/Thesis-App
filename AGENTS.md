# AGENTS.md — Thesis App Instructions for Codex

## 1. Scope

This file lives in the `thesis_app/` repository root and governs work in `thesis_system/` unless a more specific instruction file is added later.

The project is a local Streamlit research application for the BSCS thesis:

**Comparative Analysis of Markov Chain, GRU, and LSTM Algorithms for Low-Resource Sadanga Gangsa-Based Rhythmic Event Sequence Generation**

Read this file and `UX_DESIGN_SPEC.md` before changing the application.

## 2. Non-negotiable research framing

The app is a thesis/research tool, not a commercial music generator.

Use these concepts accurately:

- Sadanga Gangsa-based rhythmic-event sequences
- performance-derived rhythmic-event dataset
- verified rhythmic-event dataset
- generated rhythmic-event token sequence
- low-resource rhythmic-event sequence modeling
- sample-rendered research simulation / sound preview

Do not claim:

- that generated output is authentic Sadanga Gangsa music;
- that the system recreates a traditional performance;
- that the system represents the whole Sadanga/Sinadanga tradition;
- that rhythmic tokens are exact gong or pitch identities;
- that the models train on raw audio;
- that the sound-bank WAV files are used for algorithm training.

Defense-safe thesis claim:

> This study compares Markov Chain/N-gram, GRU, and LSTM for low-resource Sadanga Gangsa-based rhythmic-event sequence modeling.

## 3. Product goal

The interface must behave like a guided research workflow, not a flat website menu.

There are two user goals:

### Workflow A — Compare Algorithms

This is the recommended and required first workflow.

1. Prepare Data
2. Choose Settings
3. Train & Test
4. Compare Results
5. Save Results

### Workflow B — Generate & Listen

This workflow is locked until the requested algorithm comparison is genuinely complete.

1. Choose Algorithm
2. Train Final Model
3. Generate Sequence
4. Prepare Samples
5. Create & Listen
6. Save Output

Do not restore the old top navigation:

`Overview | Data Intake | Protocol | Training | Evaluation | Generation | Audio | Reports`

Do not restore a seven/eight-page menu-like experience.

## 4. UX language rule

Visible UI text must be understandable to a non-technical user.

Use plain language for:

- page/step names;
- buttons;
- status messages;
- warnings;
- empty states;
- first-level descriptions.

Keep exact research/ML terminology in:

- tooltips;
- `Technical details` expanders;
- metric guides;
- downloadable protocol/settings summaries;
- thesis-specific explanatory text where precision is necessary.

Examples:

Prefer `Train & Test` over `Formal Model-Fold Execution`.

Prefer `Prepare Data` over `Verified Event Dataset Intake`.

Prefer `Generate a Rhythm Sequence` over `Bounded Event-Sequence Generation`.

Prefer `Sound Preview` over `Sample-Rendered Rhythmic-Event Simulation` in primary UI copy, while keeping the precise term in technical detail.

## 5. Current UI architecture

Use Streamlit's modern multipage router:

- `st.Page`
- `st.navigation(..., position="hidden")`
- `st.switch_page(...)`

`app.py` is the shared application frame and router.

The custom sidebar stepper is the visible navigation system.

Do not reintroduce manual `import_module` page switching or a row of equal top-navigation buttons.

Current screen paths:

```text
src/screens/
  home.py
  compare/
    prepare_data.py
    settings.py
    train_test.py
    results.py
    export.py
  generate/
    choose_model.py
    final_training.py
    generate_sequence.py
    sound_samples.py
    listen.py
    export.py
```

Routing and workflow progression live in:

```text
src/workflows/
  routes.py
  progress.py
  guards.py
```

Shared interface building blocks live in:

```text
src/components/
  ui.py
  navigation.py
```

User-facing definitions/copy live in:

```text
src/content/
  glossary.py
  ui_text.py
```

Do not create another competing navigation/state architecture inside screen files.

## 6. Visual design direction

Preserve the dark/navy-black + teal research-tool identity, but keep it restrained and professional.

Primary design rules:

- wider desktop workspace;
- strong typographic hierarchy;
- readable body text;
- fewer bordered cards;
- one obvious primary action per step;
- secondary/technical information hidden until requested;
- compact status indicators;
- useful empty and error states;
- consistent spacing;
- clear disabled/locked states;
- accessible contrast;
- reduced-motion support;
- responsive behavior on smaller screens.

Do not repeat a large hero banner on every step.

Only Home should have the large hero treatment. Workflow pages use the compact `step_header()` pattern.

Avoid decorative UI elements that do not help the research task.

## 7. Streamlit styling rules

Use `.streamlit/config.toml` for supported theme settings first.

Use `src/styles/theme.css` for the custom design system.

Load CSS through `st.html()` from `src/styles/theme.py`.

Avoid unnecessary dependency on brittle Streamlit internal selectors. Some internal selectors are acceptable where Streamlit provides no semantic class hook, but prefer:

1. official theme settings;
2. the app's own semantic classes;
3. Streamlit widget styling only when needed.

Do not add custom JavaScript unless a required UX behavior cannot be implemented reasonably with Streamlit.

## 8. Dataset contract

Main comparison input:

`verified_event_dataset.csv`

Minimum required fields:

- `group_id`
- `event_index`
- `event_token`

Useful optional fields include:

- `onset_seconds`
- `ioi_seconds`
- `ioi_category`
- `strength_category`
- `onset_strength_norm`
- `source_id`
- `candidate_event_id`
- `clip_filename`
- `clip_path`

The UI must never expose absolute local `clip_path` values unnecessarily.

The current verified study profile expects five recording groups:

- PERF-001
- PERF-002
- PERF-003
- PERF-004
- PERF-005

Expected total event count: 586.

These counts are provenance checks, not justification for fabricating or forcing data. Unexpected values should produce warnings rather than fake corrections.

## 9. Event-token meaning

The algorithms train on ordered `event_token` sequences.

Examples include:

- START_WEAK
- START_MEDIUM
- START_STRONG
- SHORT_WEAK
- SHORT_MEDIUM
- SHORT_STRONG
- MEDIUM_WEAK
- MEDIUM_MEDIUM
- MEDIUM_STRONG
- LONG_WEAK
- LONG_MEDIUM
- LONG_STRONG

Interpretation:

- START = first event in a recording;
- SHORT / MEDIUM / LONG = timing-gap category;
- WEAK / MEDIUM / STRONG = onset-strength category.

These are not exact gong identities or N1-N9 labels.

## 10. Data preparation rules

When loading the verified dataset:

1. validate required columns;
2. exclude unusable rows in memory;
3. retain an auditable dropped-row report;
4. sort by `group_id` and `event_index`;
5. keep each recording as a separate sequence;
6. build a deterministic token vocabulary;
7. preserve token-to-ID and ID-to-token mappings.

Never create sequence windows across two recordings.

The source CSV must not be modified by the app.

## 11. Evaluation method

Use Leave-One-Recording-Out (LORO) evaluation.

With five recordings, one complete recording is held out per fold and the remaining recordings are used for training.

Never randomly split individual event rows between training and test data.

Reason: row-level splitting can leak events from the same performance into both sides of the evaluation.

The task is next-event prediction:

> Given a short window of previous rhythmic-event tokens, predict the next token.

Supported prediction window sizes remain 3, 4, and 5, with 3 as the recommended default unless the research methodology changes.

## 12. Algorithms

The comparison algorithms are:

1. Markov Chain / N-gram
2. GRU
3. LSTM

### Markov Chain

Keep it as the interpretable baseline.

- order 1 or 2;
- additive smoothing;
- unigram fallback;
- no epochs;
- no neural loss history.

### GRU / LSTM

Keep the neural models compact because the dataset is small.

Default/recommended ranges remain approximately:

- embedding dimension: 8 or 16;
- hidden units: 16 or 32;
- dropout: 0.2 to 0.4;
- batch size: 8 or 16;
- maximum epochs: 30 to 100;
- early stopping enabled;
- CPU execution supported.

Do not make the entire app fail if PyTorch is unavailable. Markov Chain must remain usable.

## 13. Evaluation result integrity

Per fold, genuine metrics may include:

- accuracy;
- macro F1;
- top-k accuracy;
- probability loss / cross-entropy style loss;
- training time;
- neural epochs completed;
- final training and validation losses for neural runs.

Never:

- fabricate result rows;
- fill failed folds with fake values;
- hard-code accuracy/F1/loss values;
- hide failures;
- claim a model completed when no genuine model artifact/result exists.

Poor model performance is valid thesis evidence.

## 14. Evaluation gating

`Generate & Listen` must stay locked unless `evaluation_run_status(state)` is genuinely complete for the currently selected algorithms and prepared recording groups.

Partial results may still be viewed/exported, but they do not unlock generation.

The Home screen must explain this dependency in plain language.

## 15. Final-model training

Final-model training is separate from LORO evaluation.

After the comparison is complete, the user may choose an evaluated algorithm and train a final generation model using all verified recording groups.

Do not report final-model training as another held-out evaluation.

Current implementation:

`src/services/generation_service.py`

The final model is stored in the active Streamlit session as `final_model_artifact`.

Changing the generation algorithm invalidates the final model and generated sequence, but the shared sample bank may be preserved because it is algorithm-independent.

## 16. Token generation

Supported bounded output lengths:

- 16 events
- 32 events
- 64 events

Generation supports:

- selected final model;
- random seed;
- temperature;
- top-k sampling;
- optional valid starting token context.

The generated output must mark its starting context separately from newly sampled events.

Do not evaluate generated sequences mainly with prediction accuracy because there is no single correct generated continuation.

## 17. Sound rendering

The sound stage uses a separate performance-derived sample bank.

Required metadata fields:

- `sample_id`
- `strength_category`
- `file_name`
- `status`

Supported strength categories:

- WEAK
- MEDIUM
- STRONG

Only accepted samples should be used by the renderer.

The algorithms do not train on WAV files.

The renderer is implemented in:

`src/services/audio_service.py`

Important timing rule:

- derive SHORT/MEDIUM/LONG interval values from the verified dataset's `ioi_seconds` values;
- do not invent arbitrary timing values if the required timing data is unavailable;
- if timing cannot be derived, block rendering and explain why.

Current renderer behavior:

- reads WAV files with SoundFile;
- converts multichannel samples to mono;
- resamples to 22,050 Hz when necessary;
- selects samples by strength category using a reproducible seed;
- places events at dataset-derived timing intervals;
- mixes overlap;
- safely limits the final peak;
- emits a WAV and token-to-sample mapping log.

Describe the result as a sound preview or sample-rendered research simulation.

## 18. Session state

The compatibility/session-state facade remains:

`src/services/session_state.py`

Do not scatter new independent readiness flags across UI files when readiness can be derived from real stored objects/results.

Prefer selectors in `src/workflows/progress.py` for UI readiness.

Important states include:

- prepared dataset;
- saved test settings;
- fold-level results;
- algorithm summary;
- neural training history;
- final model artifact;
- generated sequence;
- sample-bank metadata and WAV bytes;
- rendered audio and mapping log.

When an upstream dependency changes, invalidate only dependent downstream products.

## 19. Backend/UI separation

Keep screen files focused on user interaction and presentation.

Place computation in services/models/metrics.

Examples:

- dataset preparation → `src/services/sequence_dataset.py`
- LORO orchestration → `src/services/model_training.py`
- final training/generation → `src/services/generation_service.py`
- audio rendering → `src/services/audio_service.py`
- artifact persistence → `src/services/artifact_store.py`

Do not move algorithm logic into Streamlit screen files.

## 20. Artifact storage

Evaluation outputs continue to use versioned run folders under:

```text
results/evaluation/runs/<run_id>/
```

Do not overwrite previous evaluation runs silently.

Do not copy the user's source dataset into result folders unless explicitly required by the research design.

Downloads shown by the UI must contain genuine current-session content.

## 21. Testing

Preserve and extend pytest coverage.

Most tests must remain self-contained and use synthetic data.

Use:

```powershell
python -m pytest -m "not integration"
```

The authoritative verified-dataset integration test expects:

```text
../data_pipeline/data/verified_events/verified_event_dataset.csv
```

Do not change that external dataset as part of UI work.

When Streamlit is installed, navigation/startup tests should verify that page scripts and optional neural modules are not eagerly imported.

New backend behavior must receive tests where practical.

Existing generation and audio tests verify real Markov generation, deterministic sampling, dataset-derived timing, and valid WAV output.

## 22. Dependency rules

Core requirements are in `requirements.txt`.

PyTorch remains optional and CPU-specific through `requirements-neural-cpu.txt`.

Do not make CUDA/GPU mandatory.

SoundFile is a core dependency for the implemented sound-preview renderer.

Do not add large dependencies without a clear need.

## 23. Root repository files

Keep these at `thesis_app/`, not inside `thesis_system/`:

- `AGENTS.md`
- `UX_DESIGN_SPEC.md`
- `CHANGELOG_REDESIGN.md`
- `.gitignore`

The `.gitignore` should apply repository-wide.

## 24. Codex editing workflow

Before editing:

1. read this file;
2. read `UX_DESIGN_SPEC.md` for UI changes;
3. inspect the current implementation instead of assuming old filenames;
4. identify affected backend and UI contracts;
5. avoid unrelated rewrites.

When editing:

- preserve research correctness;
- preserve working backend behavior unless the task requires a backend change;
- prefer shared components over repeated HTML/CSS;
- prefer shared workflow selectors/guards over duplicated prerequisite logic;
- keep imports lazy around optional neural backends;
- keep code readable and defense-friendly;
- add comments only where they explain important thesis logic, not obvious Python syntax.

After editing:

1. run `python -m compileall -q thesis_system` when appropriate;
2. run relevant pytest tests;
3. run the Streamlit app locally and manually verify the changed workflow;
4. report changed files and test results;
5. never state that a runtime/UI behavior was verified if it was not actually run.

## 25. Do not regress to the old design

The following old instructions are superseded and must not be reintroduced:

- “Do not redesign the whole UI.”
- “Preserve the existing page structure.”
- “Avoid unnecessary redesign” when it prevents the guided workflow.
- “Major UI redesign is a non-goal.”
- “Generation and audio should remain only decorative/planned pages.”

A major UI/workflow redesign has been explicitly requested and is now implemented.

Future work should improve this architecture rather than reverting it.
