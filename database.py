"""Слой работы с SQL-базой данных (SQLite).

Здесь хранятся: группы, лица, биометрические признаки, проверки и журнал,
а также настройки системы (порог совпадения, режим).
"""
import sqlite3
from datetime import datetime

import config


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Создаёт таблицы из schema.sql и заполняет группы/настройки по умолчанию."""
    with open(config.SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = f.read()
    conn = get_conn()
    with conn:
        conn.executescript(schema)
        # Группы по умолчанию
        defaults = [
            (config.GROUP_OWN, "white"),
            (config.GROUP_STRANGER, "neutral"),
            (config.GROUP_BLACKLIST, "black"),
        ]
        for name, list_type in defaults:
            conn.execute(
                "INSERT OR IGNORE INTO groups (name, list_type) VALUES (?, ?)",
                (name, list_type),
            )
        # Настройки по умолчанию
        conn.execute(
            "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
            ("threshold", str(config.DEFAULT_THRESHOLD)),
        )
        conn.execute(
            "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
            ("mode", config.DEFAULT_MODE),
        )
    conn.close()


# --- Настройки ----------------------------------------------------------

def get_setting(key: str, default=None):
    conn = get_conn()
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    conn.close()
    return row["value"] if row else default


def set_setting(key: str, value) -> None:
    conn = get_conn()
    with conn:
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )
    conn.close()


# --- Группы -------------------------------------------------------------

def get_group_by_name(name: str):
    conn = get_conn()
    row = conn.execute("SELECT * FROM groups WHERE name = ?", (name,)).fetchone()
    conn.close()
    return row


def list_groups():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM groups ORDER BY id").fetchall()
    conn.close()
    return rows


# --- Лица ---------------------------------------------------------------

def add_person(full_name: str, group_id: int) -> int:
    conn = get_conn()
    with conn:
        cur = conn.execute(
            "INSERT INTO persons (full_name, group_id, status, created_at, updated_at) "
            "VALUES (?, ?, 'active', ?, ?)",
            (full_name, group_id, now(), now()),
        )
        person_id = cur.lastrowid
    conn.close()
    return person_id


def add_feature(person_id: int, embedding: bytes, source_image: str) -> None:
    conn = get_conn()
    with conn:
        conn.execute(
            "INSERT INTO face_features (person_id, embedding, source_image, created_at) "
            "VALUES (?, ?, ?, ?)",
            (person_id, embedding, source_image, now()),
        )
    conn.close()


def get_person(person_id: int):
    conn = get_conn()
    row = conn.execute(
        "SELECT p.*, g.name AS group_name, g.list_type "
        "FROM persons p JOIN groups g ON g.id = p.group_id WHERE p.id = ?",
        (person_id,),
    ).fetchone()
    conn.close()
    return row


def find_person_by_name(full_name: str):
    conn = get_conn()
    row = conn.execute(
        "SELECT p.*, g.name AS group_name, g.list_type "
        "FROM persons p JOIN groups g ON g.id = p.group_id "
        "WHERE lower(p.full_name) = lower(?) ORDER BY p.id DESC LIMIT 1",
        (full_name,),
    ).fetchone()
    conn.close()
    return row


def list_persons(include_removed: bool = True):
    conn = get_conn()
    sql = (
        "SELECT p.*, g.name AS group_name, g.list_type, "
        "(SELECT COUNT(*) FROM face_features f WHERE f.person_id = p.id) AS features "
        "FROM persons p JOIN groups g ON g.id = p.group_id"
    )
    if not include_removed:
        sql += " WHERE p.status = 'active'"
    sql += " ORDER BY p.id"
    rows = conn.execute(sql).fetchall()
    conn.close()
    return rows


def update_person_status(person_id: int, status: str) -> None:
    conn = get_conn()
    with conn:
        conn.execute(
            "UPDATE persons SET status = ?, updated_at = ? WHERE id = ?",
            (status, now(), person_id),
        )
    conn.close()


def update_person_group(person_id: int, group_id: int) -> None:
    conn = get_conn()
    with conn:
        conn.execute(
            "UPDATE persons SET group_id = ?, updated_at = ? WHERE id = ?",
            (group_id, now(), person_id),
        )
    conn.close()


# --- Признаки (эмбеддинги) ---------------------------------------------

def get_active_features():
    """Все признаки активных лиц с информацией о человеке и группе."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT f.person_id, f.embedding, p.full_name, p.status, "
        "g.name AS group_name, g.list_type "
        "FROM face_features f "
        "JOIN persons p ON p.id = f.person_id "
        "JOIN groups g ON g.id = p.group_id "
        "WHERE p.status = 'active'"
    ).fetchall()
    conn.close()
    return rows


# --- Журнал проверок ----------------------------------------------------

def add_check(image_path, mode, matched_person_id, matched_name,
              matched_group, score, decision, reason) -> int:
    conn = get_conn()
    with conn:
        cur = conn.execute(
            "INSERT INTO checks (checked_at, image_path, mode, matched_person_id, "
            "matched_name, matched_group, score, decision, reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (now(), image_path, mode, matched_person_id, matched_name,
             matched_group, score, decision, reason),
        )
        check_id = cur.lastrowid
    conn.close()
    return check_id


def list_checks(limit: int = 50):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM checks ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return rows
