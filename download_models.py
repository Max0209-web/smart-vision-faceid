"""Скачивает ONNX-модели OpenCV Zoo, если их ещё нет.

Запустите после клонирования репозитория:

    python download_models.py
"""
import os
import sys
import urllib.request

BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = [
    ("face_detection_yunet_2023mar.onnx",
     f"{BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx"),
    ("face_recognition_sface_2021dec.onnx",
     f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx"),
]

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def main():
    os.makedirs(DIR, exist_ok=True)
    for name, url in MODELS:
        dst = os.path.join(DIR, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 1000:
            print(f"  уже есть: {name}")
            continue
        print(f"  скачиваю {name} ...")
        try:
            urllib.request.urlretrieve(url, dst)
        except Exception as exc:
            print(f"  ошибка: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"    OK, {os.path.getsize(dst) // 1024} КБ")
    print("Модели готовы.")


if __name__ == "__main__":
    main()
