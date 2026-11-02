import time

import pytest

from src.utils.timing import measure


def test_measure_reports_elapsed_time_within_reasonable_bounds():
    with measure() as timer:
        time.sleep(0.05)
    assert timer.elapsed_seconds >= 0.05
    assert timer.elapsed_seconds < 2.0  # generous upper bound, avoids CI flakiness


def test_measure_still_sets_elapsed_when_the_block_raises():
    timer_ref = None
    with pytest.raises(ValueError):
        with measure() as timer:
            timer_ref = timer
            raise ValueError("boom")
    assert timer_ref.elapsed_seconds >= 0.0
