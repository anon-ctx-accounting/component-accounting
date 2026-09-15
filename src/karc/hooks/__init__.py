"""Claude Code hook adapter (M3 checkpoint c).

A best-effort telemetry channel: the hook script reads a hook payload on stdin
and records observations/events. It NEVER blocks the agent — every path exits 0
and swallows its own errors (Claude Code ignores InstructionsLoaded exit codes
entirely, and a nonzero PostToolUse exit could inject feedback into the agent;
telemetry §1.1). Registration into settings is performed by `karc init`.
"""

from karc.hooks import adapter

__all__ = ["adapter"]
