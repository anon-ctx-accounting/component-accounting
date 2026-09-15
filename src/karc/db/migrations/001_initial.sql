PRAGMA journal_mode = WAL;      -- hook 병렬 실행 하의 동시 읽기/쓰기 (telemetry §6-2)
PRAGMA foreign_keys = ON;       -- 커넥션마다 설정 (SQLite는 per-connection)

-- ===========================================================================
-- 0. meta
-- ===========================================================================
CREATE TABLE schema_migrations (
  version     INTEGER PRIMARY KEY,
  name        TEXT NOT NULL,
  applied_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- ===========================================================================
-- 1. scope / session / task
-- ===========================================================================
CREATE TABLE scopes (
  scope_id     TEXT PRIMARY KEY,                 -- ULID
  root_path    TEXT NOT NULL UNIQUE,             -- 정규화된 프로젝트 루트
  display_name TEXT,
  created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE sessions (
  session_id        TEXT PRIMARY KEY,            -- K-ARC 내부 ID
  runtime           TEXT NOT NULL,               -- 'claude-code'|'codex'|'opencode'|'hermes'|...
  native_session_id TEXT NOT NULL,               -- runtime 고유 session id
  scope_id          TEXT REFERENCES scopes(scope_id),
  model             TEXT,
  parent_session_id TEXT REFERENCES sessions(session_id),  -- Hermes compaction lineage
  transcript_path   TEXT,
  started_at        TEXT,
  ended_at          TEXT,
  UNIQUE (runtime, native_session_id)            -- 소스 3종이 같은 세션으로 수렴하는 키
);

CREATE TABLE tasks (
  task_id         TEXT PRIMARY KEY,
  session_id      TEXT NOT NULL REFERENCES sessions(session_id),
  ordinal         INTEGER NOT NULL,
  boundary_method TEXT NOT NULL CHECK (boundary_method IN
                    ('native_task','prompt_id','user_turn','window_30m')),  -- R-4/Q3
  native_task_ref TEXT,                          -- kanban task_id, prompt_id 등
  started_at      TEXT,
  ended_at        TEXT,
  UNIQUE (session_id, ordinal)
);

-- ===========================================================================
-- 2. artifact registry / identity / versions
-- ===========================================================================
CREATE TABLE artifacts (
  artifact_id        TEXT PRIMARY KEY,           -- ULID. 경로와 무관한 논리 identity
  scope_id           TEXT NOT NULL REFERENCES scopes(scope_id),
  artifact_type      TEXT NOT NULL CHECK (artifact_type IN
                       ('document','skill','rule','reference','memory','instruction','other')),
  logical_name       TEXT NOT NULL,
  canonical_path     TEXT,                       -- 현재 canonical resolved path (소실 시 NULL)
  current_version_id TEXT REFERENCES versions(version_id),
  criticality        TEXT NOT NULL DEFAULT 'normal' CHECK (criticality IN ('normal','critical')),
  pinned             INTEGER NOT NULL DEFAULT 0 CHECK (pinned IN (0,1)),
  lifecycle_state    TEXT NOT NULL DEFAULT 'active' CHECK (lifecycle_state IN
                       ('active','cold_candidate','cold','archived','invalidated')),
  owner              TEXT,
  heading_anchors    TEXT,                       -- JSON array — R-6: section은 metadata로만
  identity_confidence TEXT NOT NULL DEFAULT 'confirmed' CHECK (identity_confidence IN
                       ('confirmed','provisional')), -- §5 모호 케이스의 격리 표식
  created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  updated_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE artifact_paths (
  path_id       INTEGER PRIMARY KEY,
  artifact_id   TEXT NOT NULL REFERENCES artifacts(artifact_id),
  resolved_path TEXT NOT NULL,                   -- §6 정규화 규칙 N1~N7 적용 후
  raw_paths     TEXT,                            -- JSON: 관측된 원시 표기들 (감사용)
  status        TEXT NOT NULL CHECK (status IN ('active_canonical','active_alias','historical')),
  st_dev        INTEGER,                         -- rename 힌트 (identity 아님)
  st_ino        INTEGER,
  first_seen_at TEXT NOT NULL,
  last_seen_at  TEXT NOT NULL,
  deactivated_at TEXT,
  CHECK ((status = 'historical') = (deactivated_at IS NOT NULL))
);
-- 한 경로는 동시에 하나의 artifact에만 귀속 (복제·모호성은 §5의 절차로 처리)
CREATE UNIQUE INDEX uq_paths_active ON artifact_paths(resolved_path)
  WHERE status IN ('active_canonical','active_alias');
CREATE INDEX idx_paths_artifact ON artifact_paths(artifact_id, status);
CREATE INDEX idx_paths_inode ON artifact_paths(st_dev, st_ino) WHERE st_ino IS NOT NULL;

CREATE TABLE versions (
  version_id     TEXT PRIMARY KEY,               -- ULID
  artifact_id    TEXT NOT NULL REFERENCES artifacts(artifact_id),
  content_hash   TEXT NOT NULL,                  -- sha256(원문, 비압축)
  size_bytes     INTEGER NOT NULL,
  size_tokens    INTEGER,
  token_estimator TEXT CHECK (token_estimator IN
                    ('measured','tokenizer','bytes_div4','bytes_div2_5')), -- D8 fallback 체인 기록
  observed_at    TEXT NOT NULL,                  -- 시스템이 안 시각 (knowledge time)
  invalidated_at TEXT,                           -- 시스템이 무효를 안 시각
  valid_from     TEXT,                           -- 현실 유효 구간 (valid time, 대부분 NULL)
  valid_to       TEXT,
  superseded_by_version_id TEXT REFERENCES versions(version_id),
  supersede_reason TEXT CHECK (supersede_reason IN
                    ('new_version','correction','valid_to_expiry')),  -- R-14 폐집합
  source_channel TEXT NOT NULL,
  CHECK (superseded_by_version_id IS NULL OR superseded_by_version_id <> version_id),
  CHECK (superseded_by_version_id IS NULL OR invalidated_at IS NOT NULL),
  CHECK ((superseded_by_version_id IS NULL) = (supersede_reason IS NULL)
         OR supersede_reason IN ('correction','valid_to_expiry'))
);
CREATE INDEX idx_versions_artifact ON versions(artifact_id, observed_at);
CREATE INDEX idx_versions_hash ON versions(content_hash);
CREATE INDEX idx_versions_valid_to ON versions(valid_to)
  WHERE valid_to IS NOT NULL AND invalidated_at IS NULL;
-- artifact당 live version은 정확히 1개 (bi-temporal 위생)
CREATE UNIQUE INDEX uq_versions_live ON versions(artifact_id) WHERE invalidated_at IS NULL;

-- supersede 체인 무결성: cycle 금지 (트리거 내 재귀 CTE — §16에서 동작 검증)
CREATE TRIGGER trg_versions_no_cycle
BEFORE UPDATE OF superseded_by_version_id ON versions
WHEN NEW.superseded_by_version_id IS NOT NULL
BEGIN
  SELECT RAISE(ABORT, 'supersede cycle detected')
  WHERE EXISTS (
    WITH RECURSIVE chain(v) AS (
      SELECT NEW.superseded_by_version_id
      UNION
      SELECT vs.superseded_by_version_id
        FROM versions vs JOIN chain ON vs.version_id = chain.v
       WHERE vs.superseded_by_version_id IS NOT NULL
    )
    SELECT 1 FROM chain WHERE v = NEW.version_id
  );
END;

-- versions: bi-temporal 컬럼(invalidated_at/valid_*/superseded_by/reason)과
-- size_tokens 재추정 외에는 불변
CREATE TRIGGER trg_versions_immutable
BEFORE UPDATE ON versions
WHEN OLD.content_hash <> NEW.content_hash
  OR OLD.artifact_id <> NEW.artifact_id
  OR OLD.observed_at <> NEW.observed_at
  OR OLD.size_bytes <> NEW.size_bytes
BEGIN
  SELECT RAISE(ABORT, 'versions core fields are immutable');
END;

-- ===========================================================================
-- 3. ingestion: observations (raw, append-only) -> events (canonical)
-- ===========================================================================
CREATE TABLE ingest_observations (
  observation_id   TEXT PRIMARY KEY,      -- 결정론적: sha256(source_channel|source_native_id|보조키)
  source_channel   TEXT NOT NULL CHECK (source_channel IN
                     ('mcp','hook','transcript','otel','fswatch','manual')),
  runtime          TEXT,
  runtime_version  TEXT,
  adapter_version  TEXT NOT NULL,
  schema_version   INTEGER NOT NULL,
  source_native_id TEXT,                  -- tool_use_id | karc_call_id | transcript 위치
  observed_payload TEXT NOT NULL,         -- 원시 JSON verbatim (R-9 절제 규칙 적용 후)
  recorded_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  event_id         TEXT REFERENCES events(event_id)   -- canonicalization 결과
);
CREATE INDEX idx_obs_native ON ingest_observations(source_channel, source_native_id);
CREATE INDEX idx_obs_unassigned ON ingest_observations(recorded_at) WHERE event_id IS NULL;

-- observations: event_id 배정 외에는 불변
CREATE TRIGGER trg_obs_update_guard
BEFORE UPDATE ON ingest_observations
WHEN OLD.observed_payload <> NEW.observed_payload
  OR OLD.source_channel <> NEW.source_channel
  OR OLD.source_native_id IS NOT NEW.source_native_id
  OR OLD.recorded_at <> NEW.recorded_at
BEGIN
  SELECT RAISE(ABORT, 'ingest_observations is append-only except event_id assignment');
END;

CREATE TABLE events (
  event_id       TEXT PRIMARY KEY,               -- ULID (fold tie-break에 사용, §8)
  dedup_key      TEXT NOT NULL UNIQUE,           -- §4.2의 3-tier 결정론 키
  event_type     TEXT NOT NULL CHECK (event_type IN
                   ('discovered','loaded','read','cited','applied',
                    'validated','corrected','conflicted')),
  occurred_at    TEXT NOT NULL,                  -- 소스 timestamp (fold 순서 기준)
  recorded_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  scope_id       TEXT NOT NULL REFERENCES scopes(scope_id),   -- 이벤트 시점 denormalize
  runtime        TEXT NOT NULL,
  agent          TEXT,
  model          TEXT,
  session_id     TEXT REFERENCES sessions(session_id),
  prompt_id      TEXT,
  task_id        TEXT REFERENCES tasks(task_id),
  artifact_id    TEXT NOT NULL REFERENCES artifacts(artifact_id),
  version_id     TEXT REFERENCES versions(version_id),
  load_reason    TEXT,     -- runtime 원시값: session_start|nested_traversal|path_glob_match|include|compact|...
  load_class     TEXT CHECK (load_class IN ('preload','policy','on_demand')),  -- D2 정규화
  trigger_artifact_id TEXT REFERENCES artifacts(artifact_id), -- 탐색 경로 그래프의 원료
  token_cost     INTEGER,
  section_span   TEXT,     -- JSON (heading anchor 관측, R-6/D9)
  outcome_id     TEXT REFERENCES outcomes(outcome_id),
  confidence     TEXT NOT NULL DEFAULT 'direct' CHECK (confidence IN ('direct','approx','inferred')),
  source_channel TEXT NOT NULL,                  -- event를 생성한 1차 채널
  schema_version INTEGER NOT NULL,
  CHECK (event_type <> 'loaded' OR (load_reason IS NOT NULL AND load_class IS NOT NULL))  -- R-15
);
CREATE INDEX idx_events_scope_time ON events(scope_id, occurred_at, event_id);
CREATE INDEX idx_events_artifact_time ON events(artifact_id, occurred_at);
CREATE INDEX idx_events_artifact_type ON events(artifact_id, event_type);
CREATE INDEX idx_events_session ON events(session_id);
CREATE INDEX idx_events_task ON events(task_id) WHERE task_id IS NOT NULL;

CREATE TRIGGER trg_events_immutable
BEFORE UPDATE ON events
BEGIN
  SELECT RAISE(ABORT, 'events are immutable');
END;

CREATE TABLE outcomes (
  outcome_id     TEXT PRIMARY KEY,
  task_id        TEXT REFERENCES tasks(task_id),
  session_id     TEXT REFERENCES sessions(session_id),
  kind           TEXT NOT NULL CHECK (kind IN
                   ('user_approval','user_rejection','test_pass','test_fail',
                    'task_success','task_failure','correction')),
  occurred_at    TEXT NOT NULL,
  source_channel TEXT NOT NULL,
  confidence     TEXT NOT NULL DEFAULT 'direct' CHECK (confidence IN ('direct','approx','inferred')),
  payload        TEXT
);
CREATE INDEX idx_outcomes_task ON outcomes(task_id);

-- ===========================================================================
-- 4. usage & outcome graph
-- ===========================================================================
CREATE TABLE artifact_edges (
  edge_id        INTEGER PRIMARY KEY,
  edge_type      TEXT NOT NULL CHECK (edge_type IN
                   ('includes','references','supersedes','contradicts','derived_from',
                    'used_together','required_by','produced_by','validated_by',
                    'alias_of','duplicate_of','redirect_to')),
  src_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
  dst_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
  src_version_id TEXT REFERENCES versions(version_id),
  dst_version_id TEXT REFERENCES versions(version_id),
  observed_at    TEXT NOT NULL,
  invalidated_at TEXT,                           -- soft-invalidate (DA-12)
  created_by     TEXT NOT NULL CHECK (created_by IN ('system','policy','user')),
  evidence       TEXT,                           -- JSON: event_id 목록 + rule_id
  CHECK (src_artifact_id <> dst_artifact_id)
);
CREATE UNIQUE INDEX uq_edges_live ON artifact_edges(edge_type, src_artifact_id, dst_artifact_id)
  WHERE invalidated_at IS NULL;
CREATE INDEX idx_edges_dst ON artifact_edges(dst_artifact_id, edge_type) WHERE invalidated_at IS NULL;

-- ===========================================================================
-- 5. ARC state (materialized) + snapshots + transitions
-- ===========================================================================
CREATE TABLE arc_instances (                     -- cache-policy §5.1의 scope별 instance
  scope_id           TEXT PRIMARY KEY REFERENCES scopes(scope_id),
  c_tokens           INTEGER NOT NULL,
  c_pin_tokens       INTEGER NOT NULL DEFAULT 0,
  p_tokens           REAL NOT NULL DEFAULT 0 CHECK (p_tokens >= 0),
  config             TEXT NOT NULL,              -- JSON: ALPHA, Q_MIN, K_TAIL, W_REF, HALF_LIFE...
  policy_version     TEXT NOT NULL,              -- fold 코드 버전 (재계산 정합성 키)
  watermark_occurred_at TEXT,                    -- fold가 소비한 마지막 이벤트 시각
  watermark_event_id TEXT,
  state_hash         TEXT,                       -- 재현 검증 (FR-R2)
  updated_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE arc_entries (                       -- cache-policy §5.1의 per-artifact metadata
  scope_id      TEXT NOT NULL REFERENCES arc_instances(scope_id),
  artifact_id   TEXT NOT NULL REFERENCES artifacts(artifact_id),
  list_name     TEXT NOT NULL CHECK (list_name IN ('T1','T2','B1','B2','OVERSIZE','PINNED')),
  mru_seq       INTEGER NOT NULL,                -- scope 내 단조 증가, 클수록 MRU
  acct_tokens   INTEGER NOT NULL,                -- acct(a) = min(size_tok, ALPHA*c)
  f_score       REAL NOT NULL DEFAULT 0,         -- decayed frequency (마지막 갱신 시점 값)
  f_updated_at  TEXT,                            -- decay는 조회 시점에 lazy 적용
  n_val         INTEGER NOT NULL DEFAULT 0,
  n_corr        INTEGER NOT NULL DEFAULT 0,
  validity      TEXT NOT NULL DEFAULT 'VALID' CHECK (validity IN
                  ('VALID','SUSPECT','STALE','INVALIDATED')),
  last_ref_task TEXT,
  ghost_summary TEXT,                            -- B1/B2 ghost 요약 (<=512B 권장)
  blocked_reason TEXT,                           -- JSON: rule_id + 근거 event_id (`k-arc why`)
  PRIMARY KEY (scope_id, artifact_id)            -- I1 disjoint의 스키마 표현
);
CREATE INDEX idx_arc_entries_list ON arc_entries(scope_id, list_name, mru_seq);

CREATE TABLE arc_snapshots (
  snapshot_id        INTEGER PRIMARY KEY,
  scope_id           TEXT NOT NULL REFERENCES scopes(scope_id),
  watermark_occurred_at TEXT NOT NULL,
  watermark_event_id TEXT NOT NULL REFERENCES events(event_id),
  policy_version     TEXT NOT NULL,
  config             TEXT NOT NULL,
  p_tokens           REAL NOT NULL,
  state_hash         TEXT NOT NULL,
  created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX idx_arc_snapshots ON arc_snapshots(scope_id, watermark_occurred_at);

CREATE TABLE arc_snapshot_entries (
  snapshot_id   INTEGER NOT NULL REFERENCES arc_snapshots(snapshot_id),
  artifact_id   TEXT NOT NULL REFERENCES artifacts(artifact_id),
  list_name     TEXT NOT NULL,
  mru_seq       INTEGER NOT NULL,
  acct_tokens   INTEGER NOT NULL,
  f_score       REAL NOT NULL,
  n_val         INTEGER NOT NULL,
  n_corr        INTEGER NOT NULL,
  validity      TEXT NOT NULL,
  last_ref_task TEXT,
  PRIMARY KEY (snapshot_id, artifact_id)
);

CREATE TABLE arc_transitions (                   -- V11 설명 가능성: 모든 리스트 전이의 로그
  seq            INTEGER PRIMARY KEY,
  scope_id       TEXT NOT NULL REFERENCES scopes(scope_id),
  artifact_id    TEXT NOT NULL REFERENCES artifacts(artifact_id),
  from_list      TEXT,
  to_list        TEXT,
  rule_id        TEXT NOT NULL,                  -- 'two-distinct-task-refs', 'harmful|stale', ...
  evidence_event_id TEXT REFERENCES events(event_id),
  p_before       REAL,
  p_after        REAL,
  occurred_at    TEXT NOT NULL
);
CREATE INDEX idx_transitions_artifact ON arc_transitions(artifact_id, seq);

CREATE TRIGGER trg_transitions_immutable
BEFORE UPDATE ON arc_transitions
BEGIN SELECT RAISE(ABORT, 'arc_transitions are immutable'); END;

-- ===========================================================================
-- 6. recommendations / audit / CAS snapshot store
-- ===========================================================================
CREATE TABLE recommendations (
  rec_id       TEXT PRIMARY KEY,
  scope_id     TEXT NOT NULL REFERENCES scopes(scope_id),
  artifact_id  TEXT REFERENCES artifacts(artifact_id),
  action       TEXT NOT NULL CHECK (action IN
                 ('promote','demote','split','merge','archive','invalidate',
                  'pin','unpin','restore','cold_transition','identity_review',
                  'invalidate_or_correct','contradiction_review')),
  rule_id      TEXT NOT NULL,
  rationale    TEXT NOT NULL,            -- JSON: 구성 요소별 근거 (composite score 단독 금지)
  risk         TEXT,
  evidence     TEXT,                     -- JSON: event_id / transition seq 목록
  status       TEXT NOT NULL DEFAULT 'pending' CHECK (status IN
                 ('pending','accepted','rejected','expired','superseded','rolled_back')),
  requires_approval INTEGER NOT NULL DEFAULT 1 CHECK (requires_approval IN (0,1)),
  fs_snapshot_id INTEGER REFERENCES fs_snapshots(snapshot_id),  -- 승인 실행 전 snapshot
  created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  decided_at   TEXT,
  decided_by   TEXT,
  applied_audit_id INTEGER REFERENCES audit_log(audit_id)
);
CREATE INDEX idx_recs_pending ON recommendations(scope_id, created_at) WHERE status = 'pending';
CREATE INDEX idx_recs_artifact ON recommendations(artifact_id, created_at);

CREATE TABLE audit_log (
  audit_id    INTEGER PRIMARY KEY AUTOINCREMENT, -- AUTOINCREMENT: rowid 재사용 금지 (감사 연속성)
  at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  actor_type  TEXT NOT NULL CHECK (actor_type IN ('user','policy','system')),
  actor       TEXT,
  action      TEXT NOT NULL,
  object_type TEXT NOT NULL,
  object_id   TEXT NOT NULL,
  details     TEXT,                      -- JSON: before/after 참조 (hash/version_id)
  rollback_of INTEGER REFERENCES audit_log(audit_id)
);
CREATE TRIGGER trg_audit_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
CREATE TRIGGER trg_audit_no_delete BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;

CREATE TABLE cas_objects (
  content_hash TEXT PRIMARY KEY,         -- sha256 hex (원문 기준 — 압축 무관 identity)
  algo         TEXT NOT NULL DEFAULT 'sha256',
  size_bytes   INTEGER NOT NULL,         -- 원문 크기
  stored_size  INTEGER NOT NULL,         -- 압축 후 크기
  compression  TEXT NOT NULL CHECK (compression IN ('none','zstd')),
  storage      TEXT NOT NULL CHECK (storage IN ('db','external')),
  data         BLOB,
  external_path TEXT,                    -- objects/sha256/<2>/<62> 상대 경로
  stored_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  last_verified_at TEXT,
  CHECK ((storage = 'db') = (data IS NOT NULL)),
  CHECK ((storage = 'external') = (external_path IS NOT NULL))
);

CREATE TABLE fs_snapshots (
  snapshot_id INTEGER PRIMARY KEY,
  scope_id    TEXT NOT NULL REFERENCES scopes(scope_id),
  reason      TEXT NOT NULL,             -- 'pre_mutation'|'archive'|'pre_restore'|'manual'
  rec_id      TEXT REFERENCES recommendations(rec_id),
  created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE fs_snapshot_files (
  snapshot_id  INTEGER NOT NULL REFERENCES fs_snapshots(snapshot_id),
  artifact_id  TEXT REFERENCES artifacts(artifact_id),
  path         TEXT NOT NULL,
  content_hash TEXT NOT NULL REFERENCES cas_objects(content_hash),  -- manifest 무결성 FK
  file_mode    INTEGER,
  mtime        TEXT,
  PRIMARY KEY (snapshot_id, path)
);

-- ===========================================================================
-- 7. hard-delete 부재의 스키마 강제 (안전 원칙 10-1/10-2/10-10)
-- ===========================================================================
CREATE TRIGGER trg_no_del_artifacts BEFORE DELETE ON artifacts
BEGIN SELECT RAISE(ABORT, 'hard delete not permitted: artifacts'); END;
CREATE TRIGGER trg_no_del_versions BEFORE DELETE ON versions
BEGIN SELECT RAISE(ABORT, 'hard delete not permitted: versions'); END;
CREATE TRIGGER trg_no_del_events BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'hard delete not permitted: events'); END;
CREATE TRIGGER trg_no_del_obs BEFORE DELETE ON ingest_observations
BEGIN SELECT RAISE(ABORT, 'hard delete not permitted: ingest_observations'); END;
CREATE TRIGGER trg_no_del_outcomes BEFORE DELETE ON outcomes
BEGIN SELECT RAISE(ABORT, 'hard delete not permitted: outcomes'); END;
CREATE TRIGGER trg_no_del_edges BEFORE DELETE ON artifact_edges
BEGIN SELECT RAISE(ABORT, 'hard delete not permitted: artifact_edges'); END;
CREATE TRIGGER trg_no_del_cas BEFORE DELETE ON cas_objects
BEGIN SELECT RAISE(ABORT, 'hard delete not permitted: cas_objects'); END;
CREATE TRIGGER trg_no_del_snap BEFORE DELETE ON fs_snapshots
BEGIN SELECT RAISE(ABORT, 'hard delete not permitted: fs_snapshots'); END;
CREATE TRIGGER trg_no_del_snapfiles BEFORE DELETE ON fs_snapshot_files
BEGIN SELECT RAISE(ABORT, 'hard delete not permitted: fs_snapshot_files'); END;
CREATE TRIGGER trg_no_del_recs BEFORE DELETE ON recommendations
BEGIN SELECT RAISE(ABORT, 'hard delete not permitted: recommendations'); END;
CREATE TRIGGER trg_no_del_transitions BEFORE DELETE ON arc_transitions
BEGIN SELECT RAISE(ABORT, 'hard delete not permitted: arc_transitions'); END;
