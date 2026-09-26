# CHANGELOG_REDESIGN.md

## Revised thesis-system package

This package restructures the previous flat Streamlit dashboard into two guided research workflows and adds working final-model generation and sound-preview services.

### Navigation and UX

- Replaced the old `Overview / Data Intake / Protocol / Training / Evaluation / Generation / Audio / Reports` top-button navigation.
- Adopted `st.Page` + hidden `st.navigation` routing.
- Added a custom left workflow stepper with completed/current/locked states.
- Added Back/Continue actions and prerequisite guards.
- Rebuilt Home around introduction, definitions, workflow choice, and progress.
- Simplified visible wording for non-technical users.
- Moved detailed research terminology into help text and expanders.

### Visual system

- Reworked dark/teal theme using `.streamlit/config.toml` plus semantic CSS.
- Increased usable desktop width.
- Reduced repeated borders/cards.
- Added compact workflow headers instead of a full hero on every page.
- Improved text size, hierarchy, status treatments, empty states, and responsive behavior.
- Changed CSS loading to `st.html()`.

### Directory changes

Added:

```text
src/content/
src/screens/
  compare/
  generate/
src/workflows/
src/components/navigation.py
```

Removed the old internal `src/pages/` screen collection.

Existing model, metrics, data, and service layers remain in place to preserve backend behavior.

### Compare Algorithms

- Prepare Data: cleaner upload/validation experience and secondary technical details.
- Choose Settings: plain-language main controls plus collapsed advanced settings.
- Train & Test: focused preflight, one primary run action, compact progress, technical details collapsed.
- Compare Results: actual results first; metric documentation moved to a secondary tab.
- Save Results: task-oriented downloads and clear transition to generation.

### Generate & Listen

- Workflow is locked until complete genuine evaluation results exist.
- Added evaluated-algorithm selection.
- Added final all-recording model training service for Markov Chain, GRU, and LSTM.
- Added bounded token generation with reproducible seed, temperature, top-k, and optional starting context.
- Added performance-derived sample-bank validation flow.
- Added timing-aware WAV rendering using dataset-derived `ioi_seconds` medians.
- Added in-app audio playback and token-to-sample mapping log.
- Added generated-sequence, WAV, mapping-log, and summary downloads.

### New backend files

- `src/services/generation_service.py`
- `src/services/audio_service.py`

### Dependency change

- Added `soundfile>=0.12,<1.0` to core requirements for the implemented WAV renderer.

### Tests

Added:

- `tests/test_generation_service.py`
- `tests/test_audio_service.py`

Backend test run performed in the build environment:

```text
67 passed, 2 integration tests deselected
```

The test run excluded the two startup tests that import Streamlit because Streamlit was not installed in the build container. Python compilation of the revised project completed successfully.

Before treating the UI as fully verified, run locally with the project dependencies installed:

```powershell
streamlit run app.py
python -m pytest -m "not integration"
```

Then run the complete suite when the sibling `data_pipeline` authoritative dataset is present.

## UI Polish & Component Refinements

### Visual and Component Corrections
- **Workflow Stepper**: Added high-contrast yellow-on-black pill badge (`YOU ARE HERE`) for the active step so it never renders as a solid black circle.
- **Dropdown Internal Oblong Removal**: Stripped borders, backgrounds, and shadows from Streamlit's BaseWeb internal `<input>` inside selectboxes (`div[data-baseweb="select"] input`), eliminating the vertical capsule shape beside option labels.
- **Button Geometry**: Standardized button `border-radius` to `8px` across all sidebar navigation buttons (`Overview`, `Workflow A`, `Workflow B`), page controls, and workflow docks, utilizing full-width container layouts for sidebar items to completely eliminate all oblong/oval pill distortions.
- **Sidebar Navigation**: Replaced `width="stretch"` with `use_container_width=True` and added full-width flex card styling so sidebar buttons render as uniform, crisp square rectangles matching the app shell.
- **Control Panel 2 Collapsible**: Wrapped Sound Studio & Synthesizer controls in a collapsible expander (`expanded=False`) so the generation canvas remains clean by default.
- **Runtime Stability**: Initialized fallback generation variables (`can_generate`, `effective_seed`, `final_max_long`) and removed dangling duplicate navigation code in `choose_model.py`.

### Verification
- `python -m compileall -q thesis_system/src` (Exit code 0)
- `pytest thesis_system/tests -m "not integration"` (95 passed, 0 failures)

## Backend Bug Scan & Error Resolution

### Backend Fixes & Defensive Hardening
- **Manuscript Scorecard Generator**: Implemented `format_manuscript_scorecard()` in `src/services/artifact_store.py` and imported it in `src/screens/compare/export.py`, resolving a critical `NameError: name '_manuscript_table_text' is not defined` when saving Workflow A results. Generates LaTeX `booktabs` and Markdown scorecard tables for Chapter 4.
- **Import & Variable Resolution**: Added `from pathlib import Path` to `src/screens/compare/train_test.py` and safeguarded `dataset_metadata` construction against missing dataset attributes.
- **Sound Samples Timing Safety**: Added `require_dataset()` guard to `src/screens/generate/sound_samples.py`, initialized `intervals = None`, and guarded timing display to eliminate unbound local variable errors.
- **Sequence Generation Safety**: Added `require_dataset()` guard to `src/screens/generate/generate_sequence.py` and safeguarded `max_top_k` fallback.
- **Audio Service Validation**: Added explicit `None` check to `infer_timing_intervals()` in `src/services/audio_service.py`.
- **Results Aggregation Safety**: Guarded `aggregate_algorithm_summary()` and `_render_main_results()` in `src/screens/compare/results.py` against non-DataFrame or empty inputs.
- **Cold-Start Resilience**: Added null checks across `settings.py`, `train_test.py`, `final_training.py`, and `listen.py` so all 14 screens execute cleanly under any session initialization state.
- **Regression Tests**: Added `tests/test_manuscript_export.py` testing Markdown and LaTeX scorecard generation and defensive timing interval parameter checks.

### Verification
- `python -m compileall -q thesis_system` (Exit code 0)
- `pytest -m "not integration"` (98 passed, 0 failures)
- `pytest` full suite (100 passed, 0 failures)
- AST Symbol Resolution Check (100% resolved with 0 undefined symbols)
- Screen Execution Check (All 14 screens load with OK status)

