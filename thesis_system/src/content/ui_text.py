from __future__ import annotations

APP_NAME = "Sadanga Gangsa Rhythm Analysis and Generation System"
APP_TAGLINE = "Compare sequence models, generate rhythmic-event patterns, and create a research sound preview."

HOME_INTRO = (
    "This local research app supports a controlled comparison of Markov Chain, GRU, and LSTM "
    "using verified Sadanga Gangsa-based rhythmic-event tokens. Start by comparing the algorithms. "
    "After that comparison is complete, you can train a final model, generate a token sequence, and "
    "create a sample-rendered sound preview."
)

SAFE_SCOPE = (
    "The system studies algorithm behavior on a small performance-derived dataset. It does not train "
    "on raw audio and does not claim that generated output is an authentic Sadanga Gangsa performance."
)

COMPARE_DESCRIPTION = (
    "Load the verified data, choose the test settings, train and test all selected algorithms, then "
    "compare and save the results."
)

GENERATE_DESCRIPTION = (
    "Use the completed comparison to choose an algorithm, train one final model on all verified "
    "recordings, generate a rhythmic-event sequence, and create a sound preview from reviewed samples."
)
