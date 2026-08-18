"""Проверки простой диагностики неудачных rollout."""

from rollout_diagnostics import classify_rollout_failure


def test_classifies_cube_not_lifted():
    assert classify_rollout_failure(0.45, 0.425) == "cube not lifted"


def test_classifies_cube_dropped_after_lift():
    assert classify_rollout_failure(0.52, 0.44) == "cube dropped after lift"


def test_classifies_target_not_reached():
    assert classify_rollout_failure(0.52, 0.50) == "target not reached"