# UX_DESIGN_SPEC.md — Sadanga Gangsa System

## 1. Design intent

The app should feel like a focused desktop research tool with a guided path, not a website with many equivalent menu pages.

Primary audience:

- thesis researchers;
- panelists/advisers observing the workflow;
- users who may understand the research goal without being machine-learning specialists.

The interface should answer these questions immediately:

1. What does this app do?
2. What should I do first?
3. What step am I on?
4. What has already been completed?
5. Why is a later step locked?
6. What is the one main action on this screen?
7. Where can I find technical detail if I need it?

## 2. Information architecture

### Home

Home is an orientation and workflow-selection screen, not a dashboard full of duplicate navigation.

It contains:

- app introduction;
- clear research-scope statement;
- two workflow cards;
- current progress;
- short beginner definitions;
- technical definitions in an expander.

### Compare Algorithms

The visible stepper is:

1. Prepare Data
2. Choose Settings
3. Train & Test
4. Compare Results
5. Save Results

### Generate & Listen

The visible stepper is:

1. Choose Algorithm
2. Train Final Model
3. Generate Sequence
4. Prepare Samples
5. Create & Listen
6. Save Output

The entire Generate & Listen workflow is disabled until the requested comparison is complete.

## 3. Navigation behavior

Use the custom left sidebar as a workflow stepper.

State treatments:

- completed step → checkmark prefix;
- current step → primary/teal active treatment;
- available future/revisitable step → normal clickable treatment;
- locked step → disabled and visibly muted;
- Home → always available.

Do not use a `>` character to indicate the active page.

Do not use a row of eight equal navigation buttons.

Users should be able to revisit earlier completed steps, but changing an upstream input must invalidate dependent downstream outputs.

Back and Continue controls appear at the bottom of each step.

The primary Continue action is disabled until the current step's prerequisite product exists.

## 4. Desktop layout

Desktop is the primary target.

Use:

- expanded workflow sidebar;
- main content max width around 1300–1400 px;
- readable paragraph width around 800–900 px;
- full-width tables/charts where useful;
- generous but not excessive spacing.

Avoid the old very narrow centered content column that leaves large unused margins while squeezing tables.

## 5. Mobile/smaller-screen behavior

The layout must remain usable under 980 px.

Requirements:

- reduce horizontal page padding;
- allow Streamlit's sidebar to collapse naturally;
- stack columns when Streamlit does so;
- remove nonessential top-bar secondary text;
- avoid fixed card heights that cause clipping;
- keep all primary actions full-width when stacked.

## 6. Visual system

### Identity

Keep the existing dark research-tool identity but make it cleaner.

Base direction:

- background: near-black green/navy;
- panel background: subtle dark green-gray;
- accent: teal;
- primary text: near-white;
- secondary text: muted gray-green;
- warning: amber;
- success: green.

### Hierarchy

Only Home receives a large visual hero.

Workflow screens use:

- workflow name + Step X of Y;
- one clear H1;
- one short explanatory sentence;
- compact progress indicator.

Do not repeat Input / Process / Output cards on every screen.

### Cards

Use cards only when grouping information helps comprehension.

Avoid putting every paragraph and every number in a bordered rectangle.

Metric cards should be visually light, with a subtle accent edge rather than heavy borders and shadows.

### Borders/shadows

Use borders sparingly.

Prefer:

- subtle 1 px borders;
- restrained radius;
- little or no box shadow except Home's main hero.

## 7. Typography

Body text should normally be at least 16 px-equivalent.

Use:

- large, strong H1 for each step;
- smaller H2 section titles;
- readable body copy;
- compact technical captions.

Avoid tiny labels and helper text.

Avoid writing headings entirely in all caps except small eyebrow/section labels.

## 8. Plain-language writing style

Primary UI language should sound like a knowledgeable researcher explaining the workflow to a non-technical person.

Prefer:

- Prepare Data
- Choose Test Settings
- Train & Test
- Compare Results
- Save Results
- Choose Algorithm
- Train Final Model
- Generate Sequence
- Prepare Sound Samples
- Create & Listen
- Save Output

Avoid primary labels like:

- formal model-fold execution;
- bounded event-sequence generation;
- probabilistic evaluation protocol;
- generation configuration matrix;
- sample-rendered rhythmic-event simulation.

Those terms may appear in technical detail if necessary.

## 9. Home screen

### Hero

Contains:

- small `RESEARCH APPLICATION` eyebrow;
- `Sadanga Gangsa System` title;
- concise description of comparison, generation, and sound preview.

### Scope callout

Must say, in plain language, that:

- models use rhythmic-event tokens;
- they do not train on raw audio;
- generated output is a research simulation;
- no cultural-authenticity claim is made.

### Workflow cards

Two side-by-side cards on desktop.

#### Compare Algorithms

- visually recommended;
- explains the five steps;
- primary `Start / Continue Comparison` action.

#### Generate & Listen

- clearly locked until evaluation completes;
- disabled action while locked;
- message explains that evaluation results are used to choose an algorithm.

### Definitions

Show short cards for:

- rhythmic event;
- token;
- algorithm;
- training;
- testing/evaluation;
- generated sequence;
- sound preview.

Longer technical definitions remain collapsed.

## 10. Compare Step 1 — Prepare Data

Primary action: upload `verified_event_dataset.csv`.

After success:

- show `Data ready` status;
- show four concise metrics: source rows, usable events, recordings, token types;
- hide detailed recording/token tables in an expander;
- hide schema reference unless needed;
- replacement upload is secondary, not prominent.

Errors must tell users what needs fixing.

Warnings should not be visually equivalent to fatal errors.

Never show local absolute `clip_path` values in the main UI.

## 11. Compare Step 2 — Choose Settings

Primary action: `Save Test Settings`.

Main controls:

- algorithms;
- previous-event window size;
- top-k score setting.

Advanced settings collapsed by default:

- Markov order/smoothing;
- embedding size;
- hidden units;
- dropout;
- batch size;
- epochs;
- early stopping;
- learning rate;
- training-only validation share;
- random seed.

Show a compact setup summary after saving.

Explain LORO under a secondary `How the recording-based test works` expander.

## 12. Compare Step 3 — Train & Test

Primary action: `Start Algorithm Comparison`.

Before running, show only the necessary readiness summary:

- recording count;
- algorithms;
- number of training/test jobs;
- neural backend readiness when relevant.

During execution:

- show one progress bar;
- show current job text;
- do not flood the page with logs.

After execution:

- state how many requested jobs produced real results;
- surface failures honestly;
- keep raw job matrices/configuration under Technical details.

## 13. Compare Step 4 — Compare Results

This screen prioritizes actual results, not metric documentation.

Top level:

- comparison status;
- four small summary metrics;
- tabs.

Tabs:

1. Algorithm comparison
2. Recording-level details
3. GRU/LSTM learning curves
4. Metric guide

Algorithm comparison should show the concise per-algorithm table and an immediately useful chart such as Mean Macro F1.

Do not claim that one model is universally “best.” If highlighting the highest value, explicitly qualify it as the highest value for that metric in the current run.

The low-resource interpretation limitation must remain visible.

If evaluation is complete, show that Generate & Listen is unlocked.

## 14. Compare Step 5 — Save Results

Downloads should be task-oriented and easy to scan:

- Dataset Summary
- Saved Test Settings
- Recording-level Results
- Algorithm Summary
- Neural Training History
- Recorded Errors

Unavailable downloads stay disabled and must not contain placeholder data.

Saved filesystem locations are technical detail, not primary content.

If evaluation is complete, provide a strong next action:

`Start Generate & Listen`

## 15. Generate & Listen gating

Before a complete evaluation, every Generate & Listen sidebar step is disabled.

Home explains:

> Complete Compare Algorithms first. The evaluation results help you choose which algorithm to use for generation.

Partial evaluation does not unlock generation.

## 16. Generate Step 1 — Choose Algorithm

Show the completed comparison summary first.

Primary control: one algorithm select box.

Primary action: `Use This Algorithm`.

Changing the algorithm invalidates:

- final model;
- generated sequence;
- rendered audio.

Do not delete the validated shared sound bank solely because the algorithm changed.

## 17. Generate Step 2 — Train Final Model

Explain the conceptual separation:

- evaluation trains/tests models with held-out recordings;
- final training happens only afterward;
- final training is for generation, not new test accuracy.

Show:

- selected algorithm;
- number of recordings;
- prediction window.

Primary action:

`Train Final <Algorithm> Model`

When PyTorch is unavailable for GRU/LSTM, block the action with a clear dependency message rather than crashing the app.

## 18. Generate Step 3 — Generate Sequence

Main controls:

- sequence length;
- random seed;
- top-k;
- variation level (temperature);
- optional starting token sequence.

Use plain-language `Variation level` as the visible label and explain `temperature` in help/technical detail.

Primary action:

`Generate Sequence`

Output table must include:

- event index;
- event token;
- origin (`starting context` or `generated`).

Always keep the non-authenticity wording visible enough to prevent misinterpretation.

## 19. Generate Step 4 — Prepare Sound Samples

This step requires a generated sequence.

First show timing values derived from the verified dataset:

- SHORT median interval;
- MEDIUM median interval;
- LONG median interval.

If `ioi_seconds` is unavailable or insufficient, block rendering. Do not silently invent default timings.

Sample-bank input:

- metadata CSV;
- either individual WAV files or one WAV ZIP.

Metadata schema remains available in an expander.

Primary action:

`Check and Save Sample Bank`

Once validated, the sample bank is stored in session and should not require re-upload merely because the user moves to another wizard page.

## 20. Generate Step 5 — Create & Listen

Show compact readiness:

- generated event count;
- sound sample count;
- output format.

Primary action:

`Create Sound Preview`

After render:

- show duration;
- sample rate;
- pre-limit mix peak;
- built-in audio player;
- token-to-sample log in Technical details.

Must state that WAV samples are used after model generation and are not neural-training data.

## 21. Generate Step 6 — Save Output

Downloads:

- Generated Sequence CSV
- Sound Preview WAV
- Token-to-Sample Log CSV
- Generation Summary TXT

After successful render, show completion and provide:

- Return Home
- Review Comparison Results

## 22. Empty, locked, warning, and error states

### Empty

An empty state should explain:

1. what is missing;
2. why it is missing;
3. what action produces it.

Do not use large decorative empty boxes with no useful next step.

### Locked

Locked states should explain the prerequisite in plain language.

### Warning

Warnings are for usable-but-unexpected conditions such as provenance differences or partial job completion.

### Error

Errors are for conditions that block the requested action.

Never communicate a failed model run as success.

## 23. Progressive disclosure

Primary screens should contain the decisions/actions needed by most users.

Place these under expanders/popovers when they are not immediately required:

- data schema reference;
- dropped-row reports;
- raw job tables;
- full saved training configuration;
- confidence-interval columns;
- metric definitions;
- neural epoch histories;
- artifact filesystem paths;
- token-to-sample mapping logs.

## 24. Accessibility

Requirements:

- sufficient text/background contrast;
- readable font sizes;
- buttons labeled with actions, not ambiguous nouns;
- disabled actions accompanied by prerequisite explanations;
- do not encode state by color alone: include text/checkmarks/labels;
- honor reduced-motion preference;
- avoid unnecessary animation.

## 25. Streamlit-specific implementation guidance

Prefer built-in Streamlit widgets and layout primitives.

Use theme configuration for supported properties.

Use semantic CSS classes for custom HTML inserted by shared components.

Avoid complex custom HTML around interactive widgets.

Keep widget keys stable and descriptive.

Do not use CSS to hide required focus indicators.

Do not use unsupported DOM manipulation or injected JavaScript for navigation.

## 26. Acceptance checklist

A redesign/change is acceptable only when:

- Home clearly explains the app and both workflows;
- Compare Algorithms is visually recommended first;
- Generate & Listen cannot open before full evaluation;
- every workflow page shows Step X of Y;
- one primary action is obvious per step;
- technical settings are secondary;
- the app is wider and tables are no longer unnecessarily squeezed;
- no old eight-button top navigation remains;
- no repeated large hero + Input/Process/Output pattern remains;
- results and errors are genuine;
- cultural-authenticity claims remain prohibited;
- LORO logic remains recording-level;
- changing upstream state invalidates dependent outputs;
- generated sequences come from real final model objects;
- audio timing comes from dataset values, not invented defaults;
- sound samples are not described as model-training data;
- tests relevant to the change pass;
- the app is manually checked in Streamlit before claiming final UI verification.
