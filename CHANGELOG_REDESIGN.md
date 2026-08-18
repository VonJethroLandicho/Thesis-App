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
