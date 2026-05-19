"""Режим живой камеры: распознавание лиц "свой-чужой" в реальном времени.

Минималистичный интерфейс на русском языке. Для каждого лица в кадре —
тонкая рамка, имя, процент совпадения и решение о допуске.

Управление:
    ЛКМ по лицу — добавить это лицо в базу (имя вводится прямо в окне)
    M           — переключить режим white_list / black_list
    S           — снимок кадра в журнал
    Q / Esc     — выход
"""
import os
import time
import uuid

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
from service import FaceService

# Кадр для детекции уменьшается до этого размера по большей стороне —
# YuNet работает заметно быстрее, точность для веб-камеры не страдает.
DET_MAX = 480
# Рабочий кадр (захват/обработка/показ) ограничивается по ширине —
# гарантирует стабильный FPS независимо от разрешения веб-камеры.
WORK_WIDTH = 800
# Детекция выполняется раз в N кадров, между ними переиспользуются рамки.
DETECT_EVERY = 2

# Коды клавиш — латиница и кириллица (чтобы работало при русской раскладке).
# Физические клавиши: Q->й, M->ь, S->ы
KEYS_QUIT = {ord("q"), ord("Q"), 27, 233, 201}
KEYS_MODE = {ord("m"), ord("M"), 252, 220}
KEYS_SNAP = {ord("s"), ord("S"), 251, 219}

# Цвета в RGB (для текста через PIL)
C_OK = (95, 210, 130)
C_NO = (240, 95, 105)
C_DIM = (185, 192, 205)
C_WHITE = (245, 247, 250)
C_ACCENT = (110, 150, 255)
C_STROKE = (12, 15, 20)

GROUP_RU = {"white": "свой", "black": "чёрный список", "neutral": "чужой"}

_FONTS = {}
_FONT_FILE = None
for _f in (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf"):
    if os.path.exists(_f):
        _FONT_FILE = _f
        break


def _font(size: int):
    if size not in _FONTS:
        _FONTS[size] = (ImageFont.truetype(_FONT_FILE, size) if _FONT_FILE
                        else ImageFont.load_default())
    return _FONTS[size]


def _rgb2bgr(c):
    return (c[2], c[1], c[0])


def _key_to_char(key: int) -> str:
    """Код клавиши -> символ. Поддержка латиницы и кириллицы (cp1251)."""
    if 32 <= key <= 126:
        return chr(key)
    if 192 <= key <= 255:
        try:
            return bytes([key]).decode("cp1251")
        except Exception:
            return ""
    return ""


def render(frame, boxes, texts):
    """boxes: [(x,y,w,h,rgb)], texts: [(text,(x,y),rgb,size,anchor)]."""
    for x, y, w, h, c in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), _rgb2bgr(c), 2)
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    for text, pos, color, size, anchor in texts:
        draw.text(pos, text, font=_font(size), fill=color, anchor=anchor,
                  stroke_width=2, stroke_fill=C_STROKE)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def build_overlay(frame, faces_info, mode, enroll, fps=0.0):
    """Собирает кадр с рамками, подписями и (если нужно) панелью ввода имени."""
    boxes, texts = [], []
    for face, res in faces_info:
        allowed = res["decision"] == "ALLOWED"
        color = C_OK if allowed else C_NO
        x, y, w, h = [max(int(v), 0) for v in face[:4]]
        boxes.append((x, y, w, h, color))

        name = res["matched_name"] or "Неизвестный"
        group = GROUP_RU.get(res["matched_list_type"], "")
        head = f"{name} · {res['score']}%" if res["matched_name"] else name
        texts.append((head, (x, y - 12), color, 21, "ls"))
        if group:
            texts.append((f"группа: {group}", (x, y - 36), C_DIM, 15, "ls"))
        verdict = "Доступ разрешён" if allowed else "Доступ запрещён"
        texts.append((verdict, (x, y + h + 8), color, 18, "la"))

    fh, fw = frame.shape[:2]
    texts.append(("SMART VISION", (16, 14), C_WHITE, 19, "la"))
    texts.append((f"режим: {mode}", (16, 40), C_DIM, 15, "la"))
    texts.append((f"{fps:.0f} FPS", (fw - 16, 14), C_DIM, 15, "ra"))

    if enroll is not None:
        # Полупрозрачная панель ввода имени
        strip = frame.copy()
        cv2.rectangle(strip, (0, fh - 70), (fw, fh), (24, 28, 40), -1)
        cv2.addWeighted(strip, 0.78, frame, 0.22, 0, frame)
        texts.append(("Добавить лицо в группу «Свой»",
                      (16, fh - 56), C_ACCENT, 16, "la"))
        texts.append((f"Имя: {enroll['name']}|", (16, fh - 32), C_WHITE, 20, "la"))
        texts.append(("Enter — сохранить     Esc — отмена",
                      (fw - 16, fh - 32), C_DIM, 15, "ra"))
    else:
        texts.append(("ЛКМ по лицу — добавить в базу     M — режим     "
                      "S — снимок     Q — выход",
                      (16, fh - 14), C_DIM, 15, "ls"))
    return render(frame, boxes, texts)


def detect_fast(engine, frame):
    """Детекция лиц на уменьшенной копии кадра с возвратом координат."""
    h, w = frame.shape[:2]
    m = max(h, w)
    if m <= DET_MAX:
        return engine.detect_faces(frame)
    s = DET_MAX / m
    small = cv2.resize(frame, (int(w * s), int(h * s)))
    faces = engine.detect_faces(small)
    for f in faces:
        f[:14] = f[:14] / s  # координаты рамки и точки -> в исходный масштаб
    return faces


def _face_at(faces_info, point):
    """Возвращает лицо, в рамку которого попал клик."""
    px, py = point
    for face, _ in faces_info:
        x, y, w, h = [int(v) for v in face[:4]]
        if x <= px <= x + w and y <= py <= y + h:
            return face
    return None


def run(camera_index: int = 0, mode: str = None):
    svc = FaceService()
    mode = mode or svc.get_mode()

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Не удалось открыть камеру #{camera_index}")

    # Просим камеру отдавать кадр поменьше — это главный фактор FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except cv2.error:
        pass

    print("=" * 60)
    print(f"  Камера: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
          f"@ {cap.get(cv2.CAP_PROP_FPS):.0f}fps")
    print("  Режим живой камеры запущен")
    print("  ЛКМ по лицу — добавить | M — режим | S — снимок | Q — выход")
    print("=" * 60)

    win = "SMART VISION"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    click = {"point": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click["point"] = (x, y)

    cv2.setMouseCallback(win, on_mouse)

    enroll = None  # {"emb": ..., "name": ""} в режиме добавления лица
    fps, last = 0.0, time.time()
    faces_info = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Кадр не получен — камера отключена.")
            break

        # Ограничиваем рабочий размер кадра
        if frame.shape[1] > WORK_WIDTH:
            s = WORK_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, None, fx=s, fy=s)

        # Детекция и распознавание — не каждый кадр (экономия CPU)
        frame_idx += 1
        if frame_idx % DETECT_EVERY == 0 or not faces_info:
            faces_info = []
            for face in detect_fast(svc.engine, frame):
                try:
                    emb = svc.engine.embedding_from_face(frame, face)
                except cv2.error:
                    continue
                faces_info.append((face, svc.evaluate(emb, mode)))

        now = time.time()
        dt = now - last
        last = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

        # Клик по лицу — войти в режим добавления
        if click["point"] is not None and enroll is None:
            face = _face_at(faces_info, click["point"])
            if face is not None:
                try:
                    emb = svc.engine.embedding_from_face(frame, face)
                    enroll = {"emb": emb, "name": ""}
                except cv2.error:
                    pass
        click["point"] = None

        cv2.imshow(win, build_overlay(frame, faces_info, mode, enroll, fps))
        key = cv2.waitKey(1)
        if key == -1:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            continue
        key &= 0xFF

        if enroll is not None:
            # Режим ввода имени
            if key in (13, 10):  # Enter
                name = enroll["name"].strip()
                if name:
                    info = svc.enroll_embedding(name, enroll["emb"],
                                                config.GROUP_OWN)
                    print(f"  Добавлено: #{info['person_id']} "
                          f"{info['full_name']} -> '{info['group']}'")
                enroll = None
            elif key == 27:  # Esc
                enroll = None
            elif key in (8, 127):  # Backspace
                enroll["name"] = enroll["name"][:-1]
            else:
                enroll["name"] += _key_to_char(key)
            continue

        # Обычный режим
        if key in KEYS_QUIT:
            break
        elif key in KEYS_MODE:
            mode = "black_list" if mode == "white_list" else "white_list"
            print(f"  Режим переключён: {mode}")
        elif key in KEYS_SNAP:
            os.makedirs(config.UPLOAD_DIR, exist_ok=True)
            path = os.path.join(config.UPLOAD_DIR, f"cam_{uuid.uuid4().hex}.jpg")
            ok2, buf = cv2.imencode(".jpg", frame)
            if ok2:
                buf.tofile(path)
                res = svc.check(path, mode)
                mark = "РАЗРЕШЁН" if res["decision"] == "ALLOWED" else "ЗАПРЕЩЁН"
                print(f"  [снимок] {mark} | {res['score']}% | {res['reason']}")

        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Камера остановлена.")


if __name__ == "__main__":
    run()
