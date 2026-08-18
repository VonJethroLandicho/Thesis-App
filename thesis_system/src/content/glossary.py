from __future__ import annotations

GLOSSARY = [
    {
        "term": "Rhythmic event",
        "plain": "One detected event in a performance, described by its timing and strength.",
        "technical": "The verified dataset stores each event as an ordered token derived from timing and onset-strength features.",
    },
    {
        "term": "Token",
        "plain": "A short label the algorithms can read, such as SHORT_STRONG or LONG_WEAK.",
        "technical": "Tokens represent rhythmic timing and strength categories; they are not exact gong or pitch identities.",
    },
    {
        "term": "Algorithm",
        "plain": "A method the system uses to learn patterns and predict what event may come next.",
        "technical": "This study compares Markov Chain/N-gram, GRU, and LSTM next-event models.",
    },
    {
        "term": "Training",
        "plain": "The stage where an algorithm learns patterns from the available recording data.",
        "technical": "During evaluation, training uses only the recordings assigned to the training side of each LORO fold.",
    },
    {
        "term": "Testing / evaluation",
        "plain": "Checking how well a trained algorithm predicts events from a recording it did not train on.",
        "technical": "Leave-One-Recording-Out (LORO) holds out one complete recording for testing and uses the others for training.",
    },
    {
        "term": "Generated sequence",
        "plain": "A new ordered list of rhythmic-event tokens produced by a final trained model.",
        "technical": "Generated sequences are model outputs and are not claims of authentic traditional performance.",
    },
    {
        "term": "Sound preview",
        "plain": "An audible simulation made by matching generated token strengths to reviewed WAV samples.",
        "technical": "The algorithms do not train on these WAV files; the sample bank is used only after token generation.",
    },
]
