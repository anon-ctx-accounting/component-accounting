"""Live ARC state + recommendation synthesis (M3 checkpoint d).

Live state is computed by folding the DB's canonical events through the SAME
code path as replay (``KArcPolicy.on_event``, reached via the registry/event
loaders in ``karc.replay.trace``) — data-model §8's "state = fold(events)"
identity, so live recommendations and ``k-arc dev replay`` cannot diverge.
"""

from karc.live import state

__all__ = ["state"]
