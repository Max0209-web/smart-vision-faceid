-- Схема SQL-базы данных системы распознавания лиц "свой-чужой"
-- СУБД: SQLite

PRAGMA foreign_keys = ON;

-- Группы доступа. list_type определяет политику:
--   white   - разрешённая группа ("свой")
--   black   - чёрный список (доступ запрещён всегда)
--   neutral - прочие ("чужой")
CREATE TABLE IF NOT EXISTS groups (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT NOT NULL UNIQUE,
    list_type TEXT NOT NULL CHECK (list_type IN ('white', 'black', 'neutral'))
);

-- Люди (лица), зарегистрированные в системе
CREATE TABLE IF NOT EXISTS persons (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name  TEXT NOT NULL,
    group_id   INTEGER NOT NULL REFERENCES groups(id),
    status     TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'removed')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Биометрические признаки (эмбеддинги лиц). Один человек может иметь несколько.
CREATE TABLE IF NOT EXISTS face_features (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id    INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    embedding    BLOB NOT NULL,           -- вектор float32 (128 чисел)
    source_image TEXT,
    created_at   TEXT NOT NULL
);

-- Журнал проверок: каждое предъявленное изображение и решение системы
CREATE TABLE IF NOT EXISTS checks (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    checked_at        TEXT NOT NULL,
    image_path        TEXT NOT NULL,
    mode              TEXT NOT NULL,         -- white_list / black_list
    matched_person_id INTEGER REFERENCES persons(id),
    matched_name      TEXT,
    matched_group     TEXT,
    score             REAL NOT NULL,         -- процент совпадения 0..100
    decision          TEXT NOT NULL,         -- ALLOWED / DENIED
    reason            TEXT NOT NULL
);

-- Настройки системы (порог совпадения, режим по умолчанию)
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_features_person ON face_features(person_id);
CREATE INDEX IF NOT EXISTS idx_checks_time     ON checks(checked_at);
