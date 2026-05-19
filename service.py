"""Бизнес-логика системы "свой-чужой".

Полный цикл: добавление лица в БД, исключение/смена статуса, проверка
изображения, расчёт процента совпадения, решение о допуске и запись
события в SQL-базу и в файл-журнал.
"""
import os

import numpy as np

import config
import database as db
from face_engine import FaceEngine, FaceNotFoundError


class ServiceError(Exception):
    pass


def percent(cosine: float) -> float:
    """Косинусная близость -> процент совпадения 0..100."""
    return round(max(0.0, min(1.0, cosine)) * 100, 1)


class FaceService:
    def __init__(self):
        db.init_db()
        os.makedirs(config.LOG_DIR, exist_ok=True)
        self.engine = FaceEngine()
        self._feat_cache = None  # (матрица эмбеддингов, список записей)

    def _invalidate_cache(self) -> None:
        """Сбрасывает кэш признаков после изменения базы лиц."""
        self._feat_cache = None

    # --- Настройки ------------------------------------------------------

    def get_threshold(self) -> float:
        return float(db.get_setting("threshold", config.DEFAULT_THRESHOLD))

    def set_threshold(self, value: float) -> None:
        if not 0 < value < 1:
            raise ServiceError("Порог должен быть в диапазоне (0; 1)")
        db.set_setting("threshold", value)

    def get_mode(self) -> str:
        return db.get_setting("mode", config.DEFAULT_MODE)

    def set_mode(self, mode: str) -> None:
        if mode not in ("white_list", "black_list"):
            raise ServiceError("Режим должен быть white_list или black_list")
        db.set_setting("mode", mode)

    # --- Управление лицами ---------------------------------------------

    def enroll(self, full_name: str, image_path: str,
               group_name: str = config.GROUP_OWN) -> dict:
        """Добавляет новое лицо в указанную группу."""
        group = db.get_group_by_name(group_name)
        if group is None:
            raise ServiceError(f"Группа не найдена: {group_name}")
        embedding = self.engine.get_embedding(image_path)
        person_id = db.add_person(full_name, group["id"])
        db.add_feature(person_id, self.engine.to_blob(embedding), image_path)
        self._invalidate_cache()
        return {
            "person_id": person_id,
            "full_name": full_name,
            "group": group_name,
        }

    def add_to_blacklist(self, full_name: str, image_path: str) -> dict:
        return self.enroll(full_name, image_path, config.GROUP_BLACKLIST)

    def enroll_embedding(self, full_name: str, embedding,
                         group_name: str = config.GROUP_OWN,
                         source_image: str = "camera") -> dict:
        """Добавляет лицо по готовому эмбеддингу (используется в режиме камеры)."""
        group = db.get_group_by_name(group_name)
        if group is None:
            raise ServiceError(f"Группа не найдена: {group_name}")
        person_id = db.add_person(full_name, group["id"])
        db.add_feature(person_id, self.engine.to_blob(embedding), source_image)
        self._invalidate_cache()
        return {"person_id": person_id, "full_name": full_name, "group": group_name}

    def remove_from_own(self, person_id: int) -> None:
        """Исключение лица из группы 'свой' - перевод в группу 'чужой'."""
        person = db.get_person(person_id)
        if person is None:
            raise ServiceError(f"Лицо #{person_id} не найдено")
        stranger = db.get_group_by_name(config.GROUP_STRANGER)
        db.update_person_group(person_id, stranger["id"])
        self._invalidate_cache()

    def move_to_group(self, person_id: int, group_name: str) -> None:
        person = db.get_person(person_id)
        if person is None:
            raise ServiceError(f"Лицо #{person_id} не найдено")
        group = db.get_group_by_name(group_name)
        if group is None:
            raise ServiceError(f"Группа не найдена: {group_name}")
        db.update_person_group(person_id, group["id"])
        self._invalidate_cache()

    def set_status(self, person_id: int, status: str) -> None:
        """Полное исключение из системы (status='removed') или восстановление."""
        person = db.get_person(person_id)
        if person is None:
            raise ServiceError(f"Лицо #{person_id} не найдено")
        db.update_person_status(person_id, status)
        self._invalidate_cache()

    # --- Проверка изображения ------------------------------------------

    def _features_matrix(self):
        """Кэшированная матрица всех эмбеддингов активных лиц.

        Чтение из SQL-базы выполняется один раз, а не на каждый кадр —
        это ключевая оптимизация для режима камеры.
        """
        if self._feat_cache is None:
            rows = db.get_active_features()
            if rows:
                mat = np.stack(
                    [self.engine.from_blob(r["embedding"]) for r in rows]
                ).astype(np.float32)
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                mat = mat / norms
            else:
                mat = np.zeros((0, 128), dtype=np.float32)
            self._feat_cache = (mat, rows)
        return self._feat_cache

    def _best_match(self, embedding):
        """Ищет ближайшее лицо в БД. Возвращает (запись_признака, cosine).

        Все эмбеддинги нормированы, поэтому косинусная близость —
        это просто скалярное произведение (одно матричное умножение).
        """
        mat, rows = self._features_matrix()
        if len(rows) == 0:
            return None, -1.0
        q = np.asarray(embedding, dtype=np.float32)
        n = np.linalg.norm(q)
        if n > 0:
            q = q / n
        sims = mat @ q
        idx = int(np.argmax(sims))
        return rows[idx], float(sims[idx])

    def evaluate(self, embedding, mode: str = None) -> dict:
        """Принимает решение по готовому эмбеддингу лица (без записи в журнал).

        Используется и при проверке файла, и в режиме камеры.

        Логика:
          * чёрный список - отказ всегда (независимо от режима);
          * режим white_list - пропускаем только 'своих' выше порога;
          * режим black_list - блокируем только чёрный список,
            остальных с распознанным лицом пропускаем.
        """
        mode = mode or self.get_mode()
        threshold = self.get_threshold()

        result = {
            "mode": mode,
            "matched_person_id": None,
            "matched_name": None,
            "matched_group": None,
            "matched_list_type": None,
            "score": 0.0,
            "decision": "DENIED",
            "reason": "",
        }

        best_row, best_cos = self._best_match(embedding)
        score = percent(best_cos)
        result["score"] = score

        if best_row is None:
            result["reason"] = "База лиц пуста — сравнение невозможно"
            return result

        recognized = best_cos >= threshold
        if recognized:
            result["matched_person_id"] = best_row["person_id"]
            result["matched_name"] = best_row["full_name"]
            result["matched_group"] = best_row["group_name"]
            result["matched_list_type"] = best_row["list_type"]

        # 3. Принятие решения
        if recognized and best_row["list_type"] == "black":
            result["decision"] = "DENIED"
            result["reason"] = (
                f"Чёрный список: распознан {best_row['full_name']} "
                f"({score}%) — доступ запрещён"
            )
        elif mode == "white_list":
            if recognized and best_row["list_type"] == "white":
                result["decision"] = "ALLOWED"
                result["reason"] = (
                    f"Свой: {best_row['full_name']}, совпадение {score}% — "
                    f"доступ разрешён"
                )
            elif recognized:
                result["decision"] = "DENIED"
                result["reason"] = (
                    f"Распознан {best_row['full_name']} (группа "
                    f"'{best_row['group_name']}'), не в white list — отказ"
                )
            else:
                result["decision"] = "DENIED"
                result["reason"] = (
                    f"Неизвестное лицо: совпадение {score}% ниже порога "
                    f"{percent(threshold)}% — отказ"
                )
        else:  # black_list
            if recognized:
                result["decision"] = "ALLOWED"
                result["reason"] = (
                    f"Распознан {best_row['full_name']} ({score}%), "
                    f"не в чёрном списке — доступ разрешён"
                )
            else:
                result["decision"] = "DENIED"
                result["reason"] = (
                    f"Неизвестное лицо: совпадение {score}% ниже порога "
                    f"{percent(threshold)}% — отказ"
                )

        return result

    def check(self, image_path: str, mode: str = None) -> dict:
        """Проверяет изображение из файла и записывает событие в журнал."""
        mode = mode or self.get_mode()
        try:
            embedding = self.engine.get_embedding(image_path)
        except FaceNotFoundError:
            result = {
                "mode": mode, "matched_person_id": None, "matched_name": None,
                "matched_group": None, "matched_list_type": None, "score": 0.0,
                "decision": "DENIED",
                "reason": "Лицо на изображении не обнаружено",
            }
        else:
            result = self.evaluate(embedding, mode)
        result["image_path"] = image_path
        self._log(result)
        return result

    # --- Журналирование -------------------------------------------------

    def _log(self, result: dict) -> None:
        """Запись события в SQL-базу и в текстовый файл-журнал."""
        db.add_check(
            image_path=result["image_path"],
            mode=result["mode"],
            matched_person_id=result["matched_person_id"],
            matched_name=result["matched_name"],
            matched_group=result["matched_group"],
            score=result["score"],
            decision=result["decision"],
            reason=result["reason"],
        )
        line = (
            f"{db.now()} | файл={os.path.basename(result['image_path'])} | "
            f"лицо={result['matched_name'] or 'unknown'} | "
            f"группа={result['matched_group'] or '-'} | "
            f"совпадение={result['score']}% | "
            f"режим={result['mode']} | "
            f"решение={result['decision']} | "
            f"причина={result['reason']}\n"
        )
        with open(config.LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)

    # --- Чтение данных --------------------------------------------------

    def list_persons(self):
        return db.list_persons()

    def list_checks(self, limit: int = 50):
        return db.list_checks(limit)
