"""K-ARC snapshot store + mutation core (M3 checkpoint a).

- ``cas``   : content-addressed object store (data-model §9). zlib compression
              (the doc specifies zstd, which is not in the Python 3.12 stdlib;
              stdlib-only R-3 wins — see migration 002 and the completion report).
- ``store`` : fs_snapshots + safe mutations (archive/restore/pin/unpin) with
              audit_log, pre-mutation snapshots and atomic rename (§9.3, §10).
- ``fsck``  : integrity verification (§9.4).
"""

from karc.snapshot import cas, fsck, store

__all__ = ["cas", "fsck", "store"]
