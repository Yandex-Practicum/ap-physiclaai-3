"""Простая диагностика неудачных rollout по движению куба."""

LIFTED_CUBE_Z = 0.46
DROPPED_CUBE_Z = 0.45


def classify_rollout_failure(max_cube_z: float, final_cube_z: float) -> str:
    """Классифицировать неудачный rollout по наблюдаемому движению куба."""
    if max_cube_z < LIFTED_CUBE_Z:
        return "cube not lifted"
    if final_cube_z < DROPPED_CUBE_Z:
        return "cube dropped after lift"
    return "target not reached"