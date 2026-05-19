"""Биометрический движок: детекция лица и расчёт эмбеддинга.

Используется OpenCV DNN:
  * YuNet  - детектор лиц;
  * SFace  - извлечение 128-мерного вектора признаков (эмбеддинга).

Сравнение лиц - по косинусной близости эмбеддингов.
"""
import ctypes
import os
import shutil

import cv2
import numpy as np

import config


class FaceNotFoundError(Exception):
    """На изображении не найдено ни одного лица."""


def ascii_model_path(path: str) -> str:
    """Путь к модели без не-ASCII символов.

    OpenCV (C++) не умеет открывать ONNX-файлы по путям с кириллицей в Windows.
    Сначала пробуем короткое (8.3) имя пути, иначе копируем модель в
    гарантированно ASCII-папку.
    """
    if path.isascii() or os.name != "nt":
        return path
    # 1. Короткое (8.3) имя папки + исходное имя файла (сохраняем расширение .onnx)
    folder, fname = os.path.split(path)
    buf = ctypes.create_unicode_buffer(1024)
    n = ctypes.windll.kernel32.GetShortPathNameW(folder, buf, 1024)
    if n and buf.value and buf.value.isascii():
        candidate = os.path.join(buf.value, fname)
        if os.path.exists(candidate):
            return candidate
    # 2. Копия в ASCII-папку
    dst_dir = os.path.join(os.environ.get("PUBLIC", r"C:\Users\Public"),
                           "smartvision_models")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(path))
    if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(path):
        shutil.copy2(path, dst)
    return dst


class FaceEngine:
    def __init__(self):
        for path in (config.DET_MODEL, config.REC_MODEL):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Не найден файл модели: {path}\n"
                    f"Скачайте модели (см. README, раздел 'Установка')."
                )
        det = ascii_model_path(config.DET_MODEL)
        rec = ascii_model_path(config.REC_MODEL)
        # score_threshold=0.7, nms_threshold=0.3, top_k=5000
        self.detector = cv2.FaceDetectorYN.create(det, "", (320, 320), 0.7, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(rec, "")

    @staticmethod
    def read_image(path: str) -> np.ndarray:
        """Чтение картинки. Через imdecode - чтобы работали пути с кириллицей."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл не найден: {path}")
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Не удалось прочитать изображение: {path}")
        return img

    def detect_largest_face(self, img: np.ndarray):
        """Возвращает строку с координатами самого крупного лица или None.

        Если при стандартном пороге лицо не найдено (например, на очень
        крупном или сложном кадре), порог детектора понижается. Поскольку
        дальше берётся лицо с наибольшей площадью, ложные мелкие срабатывания
        не мешают.
        """
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))
        for score in (0.7, 0.5, 0.3):
            self.detector.setScoreThreshold(score)
            _, faces = self.detector.detect(img)
            if faces is not None and len(faces) > 0:
                self.detector.setScoreThreshold(0.7)
                return sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        self.detector.setScoreThreshold(0.7)
        return None

    def detect_faces(self, img: np.ndarray, score: float = 0.6):
        """Все найденные лица за один проход (быстро, для режима камеры)."""
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))
        self.detector.setScoreThreshold(score)
        _, faces = self.detector.detect(img)
        self.detector.setScoreThreshold(0.7)
        return [] if faces is None else list(faces)

    def embedding_from_face(self, img: np.ndarray, face) -> np.ndarray:
        """Эмбеддинг для уже найденного лица (строка из detect)."""
        aligned = self.recognizer.alignCrop(img, face)
        feature = self.recognizer.feature(aligned)
        vec = np.asarray(feature, dtype=np.float32).flatten()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def get_embedding(self, image_path: str) -> np.ndarray:
        """Извлекает 128-мерный нормализованный эмбеддинг лица из файла."""
        img = self.read_image(image_path)
        face = self.detect_largest_face(img)
        if face is None:
            raise FaceNotFoundError("Лицо на изображении не обнаружено")
        return self.embedding_from_face(img, face)

    def has_face(self, image_path: str) -> bool:
        img = self.read_image(image_path)
        return self.detect_largest_face(img) is not None

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Косинусная близость двух эмбеддингов (-1..1)."""
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    @staticmethod
    def to_blob(vec: np.ndarray) -> bytes:
        """Эмбеддинг -> bytes для хранения в BLOB."""
        return np.asarray(vec, dtype=np.float32).tobytes()

    @staticmethod
    def from_blob(blob: bytes) -> np.ndarray:
        """BLOB -> эмбеддинг."""
        return np.frombuffer(blob, dtype=np.float32)
