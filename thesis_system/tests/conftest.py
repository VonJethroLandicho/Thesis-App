from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def synthetic_event_data() -> pd.DataFrame:
    """Small unsorted recording dataset with no dependency on thesis data files."""
    return pd.DataFrame(
        {
            "group_id": [
                "PERF-002",
                "PERF-001",
                "PERF-003",
                "PERF-001",
                "PERF-002",
                "PERF-003",
                "PERF-001",
            ],
            "event_index": [2, 3, 1, 1, 1, 2, 2],
            "event_token": [
                "LONG_WEAK",
                "MEDIUM_STRONG",
                "START_STRONG",
                "START_WEAK",
                "START_MEDIUM",
                "SHORT_WEAK",
                "SHORT_MEDIUM",
            ],
        }
    )
