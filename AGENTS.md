# AGENTS.md — Thesis App System & Research Instructions

## 1. Scope & Research Framing

This document governs all agent work in `thesis_system/` for the BSCS thesis:
**Comparative Analysis of Markov Chain, GRU, and LSTM Algorithms for Low-Resource Sadanga Gangsa-Based Rhythmic Event Sequence Generation**

The application is a **local Streamlit research and evaluation platform**, not a commercial music generator.

### Mandatory Thesis Statement
> *"This study compares Markov Chain/N-gram, GRU, and LSTM algorithms for low-resource Sadanga Gangsa-based rhythmic-event sequence modeling."*

### Strict Research Boundaries (Defense Safety)
| Concept | Defense-Approved Framing | Forbidden Claim (DO NOT USE) |
| :--- | :--- | :--- |
| **System Goal** | Low-resource rhythmic-event sequence modeling | "Authentic Gangsa music generator" |
| **Tradition** | Sadanga Gangsa-based rhythmic patterns | "Recreates traditional performance" or "replaces master musicians" |
| **Input Tokens** | Discrete symbolic timing-gap + velocity categories | "Individual gong identities" (*Kalsang*, *Salaksak*) or pitches |
| **Model Training** | Trains strictly on discrete event-token sequences | "Trains on raw audio" or "trains on WAV waveforms" |
| **Audio Stage** | Sample-rendered research simulation / sound preview | "Authentic performance playback" (WAVs are purely post-gen simulation) |

---

## 2. Guided Workflow Architecture

The app enforces a linear, guided research workflow with two distinct tracks:

### Workflow A — Compare Algorithms (Mandatory First Track)
1. **Prepare Data** (`src/screens/compare/prepare_data.py`) — Validate and inspect `verified_event_dataset.csv`.
2. **Choose Settings** (`src/screens/compare/settings.py`) — Select algorithms, window size (3–5), and hyperparameters.
3. **Train & Test** (`src/screens/compare/train_test.py`) — Execute 5-fold LORO cross-validation.
4. **Compare Results** (`src/screens/compare/results.py`) — View fold metrics, confusion matrices, and statistical comparisons.
5. **Save Results** (`src/screens/compare/export.py`) — Download run artifacts, tables, and manuscript summaries.

### Workflow B — Generate & Listen (Locked Until Workflow A Completes)
*Strictly gated: locked until `evaluation_run_status(state)` is complete for selected algorithms.*
1. **Choose Algorithm** (`src/screens/generate/choose_model.py`) — Select an evaluated algorithm for final training.
2. **Train Final Model** (`src/screens/generate/final_training.py`) — Train on all 5 recording groups (not an evaluation fold).
3. **Generate Sequence** (`src/screens/generate/generate_sequence.py`) — Sample 16, 32, or 64 discrete rhythmic tokens.
4. **Prepare Samples** (`src/screens/generate/sound_samples.py`) — Audit candidate sound bank samples by velocity.
5. **Create & Listen** (`src/screens/generate/listen.py`) — Render simulation audio with dataset-derived IOI timing.
6. **Save Output** (`src/screens/generate/export.py`) — Export generated tokens, WAV audio, and timing mapping logs.

---

## 3. UI, UX, & Visual Guidelines

- **Navigation**: Uses Streamlit's multipage router (`st.Page`, `st.navigation(..., position="hidden")`, `st.switch_page(...)`). The visible stepper lives in `src/components/navigation.py`.
- **Language**: Use plain, accessible language for primary UI copy (e.g. *Train & Test*, *Prepare Data*, *Sound Preview*). Confine formal ML terms to tooltips and `Technical details` expanders.
- **Visual Identity**: Dark navy-black + teal palette loaded via `st.html()` from `src/styles/theme.py` (`theme.css`). Keep layout clean, high-contrast, scan-friendly, and accessible. No gratuitous animations or AI-cliché card grids.

---

## 4. Dataset & Event-Token Contracts

### Verified Dataset (`verified_event_dataset.csv`)
- **Required Columns**: `group_id`, `event_index`, `event_token`.
- **Optional Columns**: `onset_seconds`, `ioi_seconds`, `ioi_category`, `strength_category`, `onset_strength_norm`, `clip_filename`.
- **Provenance Standard**: Exactly 5 recording groups (`PERF-001` through `PERF-005`), 586 total verified events.
- **Immutability**: The source CSV is strictly read-only. Unusable rows are excluded in-memory with an auditable drop report.
- **Sequencing**: Sort by `group_id` and `event_index`. Never form sequence windows across two different recording groups.

### Token Interpretation
Tokens encode two orthogonal rhythmic properties:
- **Timing Gap (IOI)**: `START` (first event), `SHORT`, `MEDIUM`, `LONG`.
- **Onset Velocity**: `WEAK`, `MEDIUM`, `STRONG`.
*(Example tokens: `START_WEAK`, `SHORT_STRONG`, `MEDIUM_MEDIUM`, `LONG_STRONG`).*

---

## 5. Machine Learning & Evaluation Protocols

### Leave-One-Recording-Out (LORO) Cross-Validation
- **5 Folds**: Each fold holds out 1 complete recording (`PERF-xxx`) for testing and trains on the remaining 4 recordings.
- **Zero Leakage**: Never randomly split rows. Row-level random splitting causes intra-performance data leakage.
- **Task**: Next-event token prediction given previous $W$ tokens (window size 3, 4, or 5; default = 3).

### Algorithm Specifications
- **Markov Chain / N-gram**: Baseline model. Order 1 or 2 with additive smoothing and unigram fallback. No epochs/neural loss.
- **GRU & LSTM**: Compact PyTorch CPU networks. Embedding: 8/16; Hidden: 16/32; Dropout: 0.2–0.4; Batch: 8/16; Max Epochs: 30–100; Early stopping enabled.
- **Lazy Imports**: PyTorch imports must remain lazy. The app and Markov Chain baseline must remain fully functional if PyTorch is absent.

### Evaluation Result Integrity
- **Authentic Metrics Only**: Accuracy, Macro F1, Top-k Accuracy, Cross-Entropy Loss, Training Time, Epoch History.
- **Zero Falsification**: Never fake metrics or fill failed folds with artificial values. Poor model performance is valid research evidence.

---

## 6. Generation & Audio Simulation

### Sequence Generation
- Bounded lengths: 16, 32, or 64 events.
- Controls: Random seed, temperature, top-k sampling, optional seed prompt context.
- Distinct marking: Prompt/seed tokens must be visually distinguished from newly sampled tokens.

### Audio Simulation Rendering (`src/services/audio_service.py`)
- **Strict Timing Rule**: Inter-onset intervals (`SHORT`, `MEDIUM`, `LONG`) must be derived strictly from `ioi_seconds` in the verified dataset. If timing data is missing, fail gracefully rather than inventing arbitrary delays.
- **Audio Processing**: SoundFile mono, 22,050 Hz, normalized peaks, overlap mixing.
- **Sample Bank**: Uses accepted samples mapped to `WEAK`, `MEDIUM`, `STRONG` velocity categories.

---

## 7. Architecture & State Management

### Directory Boundaries
- `src/screens/` — UI presentation only. No heavy ML math or direct file parsing.
- `src/services/` — Core business logic (`sequence_dataset.py`, `model_training.py`, `generation_service.py`, `audio_service.py`, `artifact_store.py`).
- `src/workflows/` — Routing, guards, and readiness progression (`routes.py`, `guards.py`, `progress.py`).
- `src/components/` — Reusable Streamlit widgets and layout helpers (`ui.py`, `navigation.py`).

### Session State Invalidation Hierarchy (`src/services/session_state.py`)
- Changing Dataset $\rightarrow$ invalidates all folds, summaries, final models, generated sequences, and rendered audio.
- Changing Algorithm/Model $\rightarrow$ invalidates final model, generated sequence, and rendered audio (preserves sample bank).
- Never scatter raw state flags across screens; derive readiness through `src/workflows/progress.py`.

---

## 8. Verification & Execution Commands

When executing tasks or testing changes, use these standard commands:

```powershell
# Run unit test suite (fast non-integration tests):
.\.venv\Scripts\python.exe -m pytest thesis_system/tests -o testpaths=thesis_system/tests --basetemp=.pytest_temp -m "not integration"

# Verify Python syntax and bytecode across codebase:
.\.venv\Scripts\python.exe -m compileall -q thesis_system

# Launch Streamlit app as background process:
.\.venv\Scripts\streamlit.exe run thesis_system\app.py
```

### Fast-Path Shortcuts
- `/run app` or `/run frontend`: Start Streamlit daemon on `http://localhost:8501`.
- `/run tests` or `/test`: Execute pytest and report status.
- `/evaluate`: Run 5-fold LORO training and evaluation pipeline.
- `/status`: Run environment and dataset verification script.
