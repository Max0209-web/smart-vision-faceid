"""Пути и константы проекта."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, "models")
DET_MODEL = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
REC_MODEL = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

DB_PATH = os.path.join(BASE_DIR, "database.db")
SCHEMA_PATH = os.path.join(BASE_DIR, "schema.sql")

LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "checks.log")

DATASET_DIR = os.path.join(BASE_DIR, "faces_data", "face")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

# Группа "свой" - белый список, "Чёрный список" - чёрный, "Чужой" - нейтральная
GROUP_OWN = "Свой"
GROUP_STRANGER = "Чужой"
GROUP_BLACKLIST = "Чёрный список"

# Порог совпадения по косинусной близости эмбеддингов (0..1).
# Для модели SFace значение ~0.36 считается границей "тот же человек".
# Берём чуть выше для надёжности прототипа.
DEFAULT_THRESHOLD = 0.40

DEFAULT_MODE = "white_list"
