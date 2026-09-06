"""Back-compat alias — prefer ``mabs.queue_replay``."""

from mabs.queue_replay import batch_K_and_empty, replay_vector_commits

__all__ = ["replay_vector_commits", "batch_K_and_empty"]
