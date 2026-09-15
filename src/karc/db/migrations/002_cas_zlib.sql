-- Migration 002 — CAS compression enum: allow 'zlib'.
--
-- DEVIATION FROM data-model--mvp-schema.md §9.1 / §2 (DA-7):
--   The schema doc specifies zstd (level 3) for CAS blob compression, but
--   zstd is NOT in the Python 3.12 standard library. The M3 stdlib-only
--   constraint (R-3, brief §11) takes precedence, so the snapshot store uses
--   zlib. This migration extends cas_objects.compression to accept 'zlib'
--   (keeping 'none'/'zstd' for forward-compat and any future zstd path).
--
-- Safe rewrite: cas_objects carries no rows before M3 (the snapshot store is
-- introduced in this milestone), so the new-table-copy-rename pattern copies
-- zero rows. The doc §13.1 restricts table rewrites to derived tables; this
-- is a sanctioned, documented exception recorded in the completion report.

DROP TRIGGER trg_no_del_cas;

CREATE TABLE cas_objects_new (
  content_hash TEXT PRIMARY KEY,
  algo         TEXT NOT NULL DEFAULT 'sha256',
  size_bytes   INTEGER NOT NULL,
  stored_size  INTEGER NOT NULL,
  compression  TEXT NOT NULL CHECK (compression IN ('none','zstd','zlib')),
  storage      TEXT NOT NULL CHECK (storage IN ('db','external')),
  data         BLOB,
  external_path TEXT,
  stored_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  last_verified_at TEXT,
  CHECK ((storage = 'db') = (data IS NOT NULL)),
  CHECK ((storage = 'external') = (external_path IS NOT NULL))
);

INSERT INTO cas_objects_new SELECT * FROM cas_objects;
DROP TABLE cas_objects;
ALTER TABLE cas_objects_new RENAME TO cas_objects;

CREATE TRIGGER trg_no_del_cas BEFORE DELETE ON cas_objects
BEGIN SELECT RAISE(ABORT, 'hard delete not permitted: cas_objects'); END;
