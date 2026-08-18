from __future__ import annotations

REQUIRED_EVENT_COLUMN_NAMES = ("group_id", "event_index", "event_token")
REQUIRED_SAMPLE_COLUMN_NAMES = (
    "sample_id",
    "strength_category",
    "file_name",
    "status",
)

EVENT_COLUMN_REFERENCE = [
    {"column": "group_id", "meaning": "Complete recording or performance group used for grouped evaluation.", "required": "Yes", "example": "PERF-001"},
    {"column": "event_index", "meaning": "Sequential order of the event inside the recording.", "required": "Yes", "example": "1"},
    {
        "column": "event_token",
        "meaning": "Verified timing-and-strength rhythmic-event token used by the sequence models.",
        "required": "Yes",
        "example": "SHORT_MEDIUM",
    },
    {"column": "onset_seconds", "meaning": "Time position of the event in the source recording.", "required": "Optional", "example": "0.52"},
    {"column": "ioi_seconds", "meaning": "Inter-onset interval from the previous event.", "required": "Optional", "example": "0.34"},
    {"column": "source_id", "meaning": "Reference to the curation/source record.", "required": "Optional", "example": "KATUNOG-PERF-001"},
]

SAMPLE_BANK_COLUMN_REFERENCE = [
    {
        "column": "sample_id",
        "meaning": "Unique identifier for one performance-derived strike sample.",
        "required": "Yes",
        "example": "PERF-SAMPLE-001",
    },
    {
        "column": "strength_category",
        "meaning": "Strength class used to choose an audio-rendering sample.",
        "required": "Yes",
        "example": "STRONG",
    },
    {
        "column": "file_name",
        "meaning": "Name of the uploaded performance-derived WAV file.",
        "required": "Yes",
        "example": "strong_sample_02.wav",
    },
    {"column": "status", "meaning": "Curation status used by the renderer.", "required": "Yes", "example": "accepted"},
    {
        "column": "source_group",
        "meaning": "Optional source recording identifier retained for provenance.",
        "required": "Optional",
        "example": "PERF-003",
    },
]

ALGORITHMS = ["Markov Chain", "GRU", "LSTM"]
EXPECTED_RECORDING_GROUPS = ["PERF-001", "PERF-002", "PERF-003", "PERF-004", "PERF-005"]
EXPECTED_RECORDING_EVENT_COUNTS = {
    "PERF-001": 235,
    "PERF-002": 39,
    "PERF-003": 34,
    "PERF-004": 214,
    "PERF-005": 64,
}
EXPECTED_EVENT_COUNT = sum(EXPECTED_RECORDING_EVENT_COUNTS.values())
EXPECTED_EVENT_CLASS_COUNT = 10
DEFAULT_GENERATION_LENGTHS = [16, 32, 64]
SUPPORTED_WINDOW_SIZES = [3, 4, 5]

COMPARE_WORKFLOW_STEPS = [
    "Upload & Check Data",
    "Set Test Settings",
    "Run Training & Testing",
    "Review Algorithm Results",
    "Download Research Results",
]

GENERATE_WORKFLOW_STEPS = [
    "Choose Algorithm",
    "Train Final Model",
    "Generate Rhythm Sequence",
    "Add Sound Samples",
    "Create & Listen to Audio",
    "Download Generated Output",
]
