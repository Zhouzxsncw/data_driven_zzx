import numpy as np

from infer.policies.kemo_memory import (
    KEMO_ACTION_PERM,
    KemoMemory,
    inverse_reorder_kemo_action,
    reorder_kemo_action,
)


def test_memory_reset_and_bounded_history():
    memory = KemoMemory(max_length=2)
    memory.append({"state": np.zeros(16, dtype=np.float32)})
    second = memory.append({"state": np.ones(16, dtype=np.float32)})
    third = memory.append({"state": np.full(16, 2, dtype=np.float32)})
    assert second["kemo_memory"]["length"] == 2
    assert third["kemo_memory"]["length"] == 2
    assert third["kemo_memory"]["episode_step"] == 3
    assert memory.length == 2
    memory.reset()
    assert memory.length == 0 and memory.episode_step == 0


def test_action_layout_is_reversible():
    raw = np.arange(16, dtype=np.float32)
    assert np.array_equal(reorder_kemo_action(raw), raw[list(KEMO_ACTION_PERM)])
    assert np.array_equal(inverse_reorder_kemo_action(reorder_kemo_action(raw)), raw)
