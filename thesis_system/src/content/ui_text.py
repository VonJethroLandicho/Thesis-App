from __future__ import annotations

APP_NAME = "Sadanga Gangsa Rhythm Analysis and Generation System"
APP_TAGLINE = "Compare AI rhythm models, generate new gong strike patterns, and hear a research sound preview."

HOME_INTRO = (
    "This research platform lets you compare three sequence models (Markov Chain, GRU, and LSTM) "
    "on real Sadanga Gangsa gong rhythm data. In Workflow A, you'll run a 5-round fair comparison "
    "to see which algorithm predicts rhythm patterns best. Once that evaluation is finished, Workflow B "
    "unlocks so you can train a final model, generate original rhythm strike sequences, and hear a realistic "
    "sound preview made from recorded gong strike samples."
)

SAFE_SCOPE = (
    "This study evaluates sequence prediction algorithms on a compact, 586-strike rhythm dataset. "
    "The AI models train strictly on discrete symbolic rhythm tokens (timing pauses and strike strengths), "
    "not raw audio waveforms. Audio previews are research simulations created by matching generated tokens "
    "to real gong strike sound clips; they do not claim to recreate or replace authentic traditional performances."
)

COMPARE_DESCRIPTION = (
    "Upload the verified gong strike data, configure your shared test settings, run the 5-round fair "
    "evaluation across Markov Chain, GRU, and LSTM, and compare the accuracy and error scores."
)

GENERATE_DESCRIPTION = (
    "Select an evaluated algorithm, train a final model on all 5 recording groups, generate original "
    "rhythm strike sequences (16, 32, or 64 strikes), and listen to an audible sound simulation."
)
