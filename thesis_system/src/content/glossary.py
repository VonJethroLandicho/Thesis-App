from __future__ import annotations

GLOSSARY = [
    {
        "term": "Rhythmic event (Gong strike)",
        "plain": "One single detected gong mallet strike in a recorded performance, described by when it happened and how hard it was struck.",
        "technical": "A discrete symbolic event extracted from acoustic recordings, parameterized by inter-onset interval (IOI) and normalized onset energy.",
    },
    {
        "term": "Event token",
        "plain": "A short code label the AI reads (such as SHORT_STRONG or LONG_WEAK) combining the timing pause before the strike and its strike strength.",
        "technical": "A discrete categorical symbol in the model vocabulary representing a joint (timing_gap, strike_velocity) state; not a pitch or gong identity.",
    },
    {
        "term": "Timing gap (IOI / Inter-Onset Interval)",
        "plain": "The pause or time gap (in seconds) between two consecutive gong strikes (categorized into START, SHORT, MEDIUM, or LONG).",
        "technical": "The temporal distance in seconds between consecutive acoustic onset detections: START (initial event), SHORT (<0.5s), MEDIUM (0.5–1.2s), LONG (>1.2s).",
    },
    {
        "term": "Strike velocity (Onset strength)",
        "plain": "How hard or soft a gong was struck (categorized into WEAK, MEDIUM, or STRONG strike dynamics).",
        "technical": "RMS energy and spectral flux magnitude at onset time, normalized and partitioned into discrete amplitude bins.",
    },
    {
        "term": "Algorithm / Sequence model",
        "plain": "The machine learning method that listens to past strikes and tries to predict what strike comes next in the rhythm.",
        "technical": "A discrete sequence modeling architecture mapping a context window of prior tokens to a categorical probability distribution over next tokens.",
    },
    {
        "term": "Markov Chain (Baseline pattern counter)",
        "plain": "A fast statistical model that counts how often specific strike combinations happened in past recordings to guess the next strike.",
        "technical": "An n-gram transition matrix model (order 1 or 2) with Laplace/additive smoothing and unigram backoff, trained by direct frequency counting.",
    },
    {
        "term": "GRU (Lightweight neural memory)",
        "plain": "A compact neural network that maintains a running internal memory of recent rhythm flow to guess upcoming strikes.",
        "technical": "A Gated Recurrent Unit network utilizing reset and update gates to maintain hidden continuous state representations without separate cell states.",
    },
    {
        "term": "LSTM (Deep recurrent memory)",
        "plain": "A neural network with specialized memory gates that can remember motifs over longer stretches of rhythm.",
        "technical": "A Long Short-Term Memory network utilizing input, forget, and output gates with dedicated cell states to capture long-range temporal dependencies.",
    },
    {
        "term": "5-Round fair testing (LORO Cross-Validation)",
        "plain": "A strict evaluation method where the AI learns from 4 recordings and is tested on the 1 held-out recording, repeating 5 times so every recording is tested without the AI ever seeing it during training.",
        "technical": "Leave-One-Recording-Out (LORO) cross-validation where each of the 5 recording groups is held out as the test set while the remaining 4 serve as training data, preventing intra-performance leakage.",
    },
    {
        "term": "Memory window (Context size W)",
        "plain": "How many prior gong strikes the model listens to before guessing the next strike (typically 3 strikes).",
        "technical": "The context window length W in {3, 4, 5} defining the conditioning history x_(t-W:t-1) for next-token prediction p(x_t | x_(t-W:t-1)).",
    },
    {
        "term": "Exact first-guess accuracy",
        "plain": "The percentage of times the model's single #1 guess was the exact right next gong strike.",
        "technical": "Top-1 categorical accuracy measuring exact argmax prediction matches against held-out ground truth tokens.",
    },
    {
        "term": "Top-3 candidate likelihood (Top-k recall)",
        "plain": "Checks if the true strike was among the model's top 3 best guesses. In ensemble music, multiple rhythmic continuations are often musically valid.",
        "technical": "Top-k categorical accuracy evaluating whether the true target token falls within the model's k highest probability predictions.",
    },
    {
        "term": "Rhythm balance score (Macro F1)",
        "plain": "Measures overall fairness across all rhythm types. It ensures the model is penalized if it only guesses common strikes while ignoring rare strikes.",
        "technical": "Unweighted arithmetic mean of F1 scores across all vocabulary classes, treating rare tokens (e.g., START_WEAK) with equal weight to common tokens.",
    },
    {
        "term": "Mistake & uncertainty rate (Cross-Entropy loss)",
        "plain": "Measures how far off or uncertain the model's probability predictions were. Lower loss means higher confidence and accuracy.",
        "technical": "Mean cross-entropy loss over held-out evaluation tokens, quantifying the negative log-likelihood of target tokens under the model's predicted distribution.",
    },
    {
        "term": "Creativity / Randomness (Temperature)",
        "plain": "Controls how adventurous the model is during sequence generation. Lower values produce safer, standard rhythms; higher values introduce surprising variations.",
        "technical": "A scaling factor T applied to model logits prior to softmax; T < 1.0 sharpens the distribution toward the mode, while T > 1.0 flattens it toward uniform sampling.",
    },
    {
        "term": "Candidate pool size (Top-k cutoff)",
        "plain": "Restricts the model's choices to only its top k most likely strikes, cutting out awkward or musically implausible transitions.",
        "technical": "Top-k truncation filtering that zeros out log-probabilities outside the top k candidates before renormalizing the sampling distribution.",
    },
    {
        "term": "Anti-repetition control (Repetition penalty)",
        "plain": "Prevents the model from getting stuck in an unnatural loop repeating the exact same gong strike over and over.",
        "technical": "A multiplicative discount applied to the logits of recently generated tokens to discourage immediate consecutive repetition.",
    },
    {
        "term": "Strike flow validity (2-strike pairs)",
        "plain": "The percentage of two-strike pairs generated by the model that match valid strike transitions seen in authentic Sadanga Gangsa recordings.",
        "technical": "Bigram transition validity measuring the fraction of generated consecutive token pairs (x_t, x_(t+1)) present in the training transition matrix.",
    },
    {
        "term": "Phrasing novelty (4-strike patterns)",
        "plain": "The percentage of 4-strike rhythm patterns newly composed by the model rather than copied directly from the original recordings.",
        "technical": "4-gram sequence novelty measuring the proportion of generated 4-token sub-sequences that do not appear in the training corpus.",
    },
    {
        "term": "Originality check (Copy detector)",
        "plain": "Confirms whether the generated rhythm is a novel composition or an accidental word-for-word copy of an existing recording.",
        "technical": "Verbatim sub-sequence match detector checking for full-sequence or long-span identity against any training recording.",
    },
    {
        "term": "Sound preview (Research simulation)",
        "plain": "An audible preview created after rhythm generation by placing reviewed gong strike WAV audio samples at dataset-calculated timing gaps.",
        "technical": "A sample-rendered post-generation research audio simulation; the AI models train solely on symbolic tokens, never on raw audio waveforms.",
    },
]
