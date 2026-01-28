PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS episodes (
  episode_id TEXT PRIMARY KEY,
  date TEXT,
  brief TEXT,
  duration_sec REAL,
  status TEXT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS artifacts (
  artifact_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  uri TEXT NOT NULL,
  hash TEXT NOT NULL,
  tags_json TEXT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS critiques (
  critique_id TEXT PRIMARY KEY,
  episode_id TEXT,
  gate TEXT,
  labels_json TEXT,
  notes TEXT,
  accepted INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
);

CREATE TABLE IF NOT EXISTS character_bible_versions (
  name TEXT NOT NULL,
  version INTEGER NOT NULL,
  json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  PRIMARY KEY (name, version)
);

CREATE TABLE IF NOT EXISTS retrieval_notes (
  note_id TEXT PRIMARY KEY,
  type TEXT NOT NULL,
  text TEXT NOT NULL,
  tags_json TEXT,
  episode_id TEXT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS retrieval_fts USING fts5(
  note_id,
  type,
  text,
  tags,
  episode_id,
  content=''
);

CREATE TRIGGER IF NOT EXISTS retrieval_notes_ai
AFTER INSERT ON retrieval_notes
BEGIN
  INSERT INTO retrieval_fts(note_id, type, text, tags, episode_id)
  VALUES (new.note_id, new.type, new.text, new.tags_json, new.episode_id);
END;

CREATE TRIGGER IF NOT EXISTS retrieval_notes_ad
AFTER DELETE ON retrieval_notes
BEGIN
  INSERT INTO retrieval_fts(retrieval_fts, note_id, type, text, tags, episode_id)
  VALUES ('delete', old.note_id, old.type, old.text, old.tags_json, old.episode_id);
END;

CREATE TRIGGER IF NOT EXISTS retrieval_notes_au
AFTER UPDATE ON retrieval_notes
BEGIN
  INSERT INTO retrieval_fts(retrieval_fts, note_id, type, text, tags, episode_id)
  VALUES ('delete', old.note_id, old.type, old.text, old.tags_json, old.episode_id);
  INSERT INTO retrieval_fts(note_id, type, text, tags, episode_id)
  VALUES (new.note_id, new.type, new.text, new.tags_json, new.episode_id);
END;
