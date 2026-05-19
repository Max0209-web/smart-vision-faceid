"""Веб-интерфейс системы "свой-чужой" на стандартной библиотеке Python.

Без сторонних веб-фреймворков: http.server. Позволяет на защите
наглядно показать добавление лица, исключение, проверку изображения,
процент совпадения, режимы white/black list и журнал.
"""
import base64
import json
import mimetypes
import os
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

import config
from service import FaceService, ServiceError

SVC = None  # инициализируется в run()


# --- Разбор multipart/form-data (для загрузки файлов) -------------------

def parse_multipart(body: bytes, boundary: bytes) -> dict:
    """Простой разбор multipart/form-data. Возвращает {имя: значение|bytes}."""
    result = {}
    delim = b"--" + boundary
    for part in body.split(delim):
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        if b"\r\n\r\n" not in part:
            continue
        raw_headers, content = part.split(b"\r\n\r\n", 1)
        headers = raw_headers.decode("utf-8", "ignore")
        name, filename = None, None
        for line in headers.split("\r\n"):
            if line.lower().startswith("content-disposition"):
                for token in line.split(";"):
                    token = token.strip()
                    if token.startswith('name="'):
                        name = token[6:-1]
                    elif token.startswith('filename="'):
                        filename = token[10:-1]
        if name is None:
            continue
        if filename is not None:
            result[name] = {"filename": filename, "data": content}
        else:
            result[name] = content.decode("utf-8", "ignore")
    return result


# --- Сериализация записей БД -------------------------------------------

def person_dict(r):
    return {
        "id": r["id"], "name": r["full_name"], "group": r["group_name"],
        "list_type": r["list_type"], "status": r["status"],
        "features": r["features"],
    }


def check_dict(r):
    return {
        "id": r["id"], "time": r["checked_at"], "score": r["score"],
        "decision": r["decision"], "reason": r["reason"],
        "name": r["matched_name"], "group": r["matched_group"],
        "mode": r["mode"], "image": r["image_path"],
    }


def list_samples():
    """Список тестовых изображений из faces_data/face."""
    samples = []
    if not os.path.isdir(config.DATASET_DIR):
        return samples
    for folder in sorted(os.listdir(config.DATASET_DIR)):
        d = os.path.join(config.DATASET_DIR, folder)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                samples.append({
                    "name": folder.replace("_", " ").title(),
                    "path": os.path.join(d, f),
                })
                break
    return samples


def decode_b64_image(data_url: str) -> np.ndarray:
    """base64 data-URL из браузера -> изображение OpenCV."""
    raw = data_url.split(",", 1)[-1]
    buf = np.frombuffer(base64.b64decode(raw), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ServiceError("Не удалось декодировать кадр")
    return img


def save_b64_image(data_url: str) -> str:
    """Сохраняет кадр из браузера в файл, возвращает путь."""
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    raw = data_url.split(",", 1)[-1]
    path = os.path.join(config.UPLOAD_DIR, f"{uuid.uuid4().hex}.jpg")
    with open(path, "wb") as f:
        f.write(base64.b64decode(raw))
    return path


def safe_path(path: str) -> str:
    """Разрешает отдавать только файлы из dataset и uploads."""
    real = os.path.realpath(path)
    allowed = [os.path.realpath(config.DATASET_DIR),
               os.path.realpath(config.UPLOAD_DIR)]
    if any(real.startswith(a) for a in allowed) and os.path.isfile(real):
        return real
    raise ServiceError("Доступ к файлу запрещён")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # тихий режим

    # --- Отправка ответов ----------------------------------------------

    def send_json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_html(self, html):
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    def get_payload(self) -> dict:
        """Возвращает данные запроса: JSON или multipart."""
        ctype = self.headers.get("Content-Type", "")
        body = self.read_body()
        if ctype.startswith("multipart/form-data"):
            boundary = ctype.split("boundary=")[-1].encode()
            return parse_multipart(body, boundary)
        if body:
            return json.loads(body.decode("utf-8"))
        return {}

    def resolve_image(self, payload) -> str:
        """Возвращает путь к изображению: из dataset или из загруженного файла."""
        if "image" in payload and isinstance(payload["image"], dict):
            os.makedirs(config.UPLOAD_DIR, exist_ok=True)
            up = payload["image"]
            ext = os.path.splitext(up["filename"])[1] or ".jpg"
            fname = f"{uuid.uuid4().hex}{ext}"
            path = os.path.join(config.UPLOAD_DIR, fname)
            with open(path, "wb") as f:
                f.write(up["data"])
            return path
        if payload.get("image_b64"):
            return save_b64_image(payload["image_b64"])
        if payload.get("sample_path"):
            return payload["sample_path"]
        raise ServiceError("Изображение не передано")

    # --- GET -----------------------------------------------------------

    def do_GET(self):
        try:
            if self.path == "/" or self.path.startswith("/index"):
                self.send_html(PAGE)
            elif self.path == "/api/state":
                self.send_json(self.state())
            elif self.path == "/api/samples":
                self.send_json({"samples": list_samples()})
            elif self.path.startswith("/img?"):
                self.serve_image()
            else:
                self.send_json({"error": "not found"}, 404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, 400)

    def serve_image(self):
        from urllib.parse import urlparse, parse_qs, unquote
        qs = parse_qs(urlparse(self.path).query)
        path = unquote(qs.get("path", [""])[0])
        real = safe_path(path)
        ctype = mimetypes.guess_type(real)[0] or "application/octet-stream"
        with open(real, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def state(self):
        return {
            "persons": [person_dict(r) for r in SVC.list_persons()],
            "checks": [check_dict(r) for r in SVC.list_checks(30)],
            "groups": [g["name"] for g in __import__("database").list_groups()],
            "mode": SVC.get_mode(),
            "threshold": SVC.get_threshold(),
        }

    # --- POST ----------------------------------------------------------

    def do_POST(self):
        try:
            payload = self.get_payload()
            route = self.path

            if route == "/api/recognize":
                # Покадровое распознавание для живой камеры (без журнала)
                img = decode_b64_image(payload["image_b64"])
                mode = payload.get("mode") or SVC.get_mode()
                faces = []
                for face in SVC.engine.detect_faces(img):
                    try:
                        emb = SVC.engine.embedding_from_face(img, face)
                    except cv2.error:
                        continue
                    res = SVC.evaluate(emb, mode)
                    x, y, w, h = [int(v) for v in face[:4]]
                    faces.append({
                        "box": [x, y, w, h],
                        "name": res["matched_name"],
                        "score": res["score"],
                        "decision": res["decision"],
                        "list_type": res["matched_list_type"],
                        "reason": res["reason"],
                    })
                self.send_json({"faces": faces,
                                "w": img.shape[1], "h": img.shape[0]})

            elif route == "/api/check":
                image = self.resolve_image(payload)
                mode = payload.get("mode") or SVC.get_mode()
                result = SVC.check(image, mode)
                self.send_json({"result": result, "state": self.state()})

            elif route == "/api/enroll":
                image = self.resolve_image(payload)
                name = (payload.get("name") or "").strip()
                group = payload.get("group") or config.GROUP_OWN
                if not name:
                    raise ServiceError("Не указано имя")
                info = SVC.enroll(name, image, group)
                self.send_json({"info": info, "state": self.state()})

            elif route == "/api/remove":
                SVC.remove_from_own(int(payload["id"]))
                self.send_json({"state": self.state()})

            elif route == "/api/move":
                SVC.move_to_group(int(payload["id"]), payload["group"])
                self.send_json({"state": self.state()})

            elif route == "/api/status":
                SVC.set_status(int(payload["id"]), payload["value"])
                self.send_json({"state": self.state()})

            elif route == "/api/mode":
                SVC.set_mode(payload["mode"])
                self.send_json({"state": self.state()})

            elif route == "/api/threshold":
                SVC.set_threshold(float(payload["value"]))
                self.send_json({"state": self.state()})

            else:
                self.send_json({"error": "not found"}, 404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, 400)


def run(host="127.0.0.1", port=8000, certfile=None, keyfile=None):
    global SVC
    print("Инициализация сервиса (загрузка моделей)...")
    SVC = FaceService()
    server = ThreadingHTTPServer((host, port), Handler)
    scheme = "http"
    if certfile and keyfile:
        import ssl
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile, keyfile)
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    print("=" * 60)
    print(f"  Веб-интерфейс запущен:  {scheme}://{host}:{port}")
    print("  Остановка: Ctrl+C")
    print("=" * 60)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлено.")
        server.server_close()


# --- HTML-страница (встроена для простоты запуска) ----------------------

PAGE = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SMART VISION — свой / чужой</title>
<style>
  :root{
    --bg:#1b212e; --surface:#232b3c; --surface2:#2c3650; --line:#333d54;
    --txt:#e7ecf4; --muted:#8993a8; --accent:#5b8cff;
    --ok:#4ecb88; --no:#fa5f6b;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--txt);
       font-family:'Segoe UI',Roboto,system-ui,sans-serif;font-size:14px;
       line-height:1.5;padding:28px max(28px,4vw);max-width:1240px;margin:0 auto}

  /* --- Шапка --- */
  header{display:flex;justify-content:space-between;align-items:center;
         gap:20px;flex-wrap:wrap;margin-bottom:24px}
  .brand{display:flex;align-items:center;gap:13px}
  .logo{width:42px;height:42px;border-radius:11px;background:var(--accent);
        display:flex;align-items:center;justify-content:center;
        font-weight:800;font-size:16px;color:#fff;letter-spacing:.5px}
  .title{font-size:18px;font-weight:700;letter-spacing:.2px}
  .sub{font-size:12px;color:var(--muted)}
  .controls{display:flex;align-items:center;gap:18px;flex-wrap:wrap}

  /* сегментированный переключатель режима */
  .seg{display:flex;background:var(--surface);border-radius:10px;padding:4px;
       gap:4px}
  .seg button{border:0;background:transparent;color:var(--muted);
       padding:7px 14px;border-radius:7px;font-weight:600;font-size:13px;
       cursor:pointer;transition:.15s}
  .seg button.active{background:var(--accent);color:#fff}
  .seg button:not(.active):hover{color:var(--txt)}

  .thr{display:flex;align-items:center;gap:9px;font-size:13px;color:var(--muted)}
  .thr input[type=range]{width:120px;accent-color:var(--accent);cursor:pointer}
  .thrval{color:var(--txt);font-weight:600;min-width:38px}

  .modehint{background:var(--surface);border-left:3px solid var(--accent);
       border-radius:8px;padding:9px 14px;font-size:12.5px;color:var(--muted);
       margin-bottom:22px}

  /* --- Карточки --- */
  .card{background:var(--surface);border-radius:16px;padding:20px;margin-bottom:20px}
  .card-h{font-size:13px;font-weight:700;letter-spacing:.6px;text-transform:uppercase;
          color:var(--muted);margin-bottom:14px}
  .card-h b{color:var(--txt)}

  /* --- Камера --- */
  .camera{display:grid;grid-template-columns:1.4fr 1fr;gap:20px}
  @media(max-width:780px){.camera{grid-template-columns:1fr}}
  .camwrap{position:relative;background:#11151e;border-radius:14px;
           overflow:hidden;aspect-ratio:4/3}
  .camwrap video{display:block;width:100%;height:100%;object-fit:cover}
  .camwrap canvas{position:absolute;inset:0;width:100%;height:100%}
  .camwrap.live canvas{cursor:pointer}
  .camhint{position:absolute;inset:0;display:flex;align-items:center;
           justify-content:center;text-align:center;color:var(--muted);
           font-size:13px;flex-direction:column;gap:8px}
  .camhint .big{font-size:34px;opacity:.5}
  .campanel{display:flex;flex-direction:column;gap:14px}

  /* --- Блок вердикта --- */
  .verdict{flex:1;border-radius:14px;padding:18px;background:var(--surface2);
           border-left:4px solid var(--line);display:flex;flex-direction:column;
           justify-content:center;min-height:150px;transition:.2s}
  .verdict.allowed{border-left-color:var(--ok);
                   background:linear-gradient(120deg,rgba(78,203,136,.12),var(--surface2))}
  .verdict.denied{border-left-color:var(--no);
                  background:linear-gradient(120deg,rgba(250,95,107,.12),var(--surface2))}
  .v-title{font-size:21px;font-weight:800;letter-spacing:.3px}
  .verdict.allowed .v-title{color:var(--ok)}
  .verdict.denied .v-title{color:var(--no)}
  .verdict.idle .v-title{color:var(--muted)}
  .v-reason{font-size:13px;color:var(--txt);margin-top:5px}
  .v-bar{height:8px;border-radius:5px;background:rgba(0,0,0,.28);
         overflow:hidden;margin:12px 0 7px}
  .v-bar>div{height:100%;width:0;background:var(--accent);transition:.3s}
  .v-meta{font-size:12px;color:var(--muted)}

  /* --- Кнопки --- */
  button{font:inherit;border:0;border-radius:9px;background:var(--accent);
         color:#fff;font-weight:600;padding:10px 16px;cursor:pointer;transition:.15s}
  button:hover:not(:disabled){filter:brightness(1.12)}
  button:disabled{opacity:.4;cursor:default}
  button.ghost{background:var(--surface2);color:var(--txt)}
  button.sm{padding:6px 10px;font-size:12px;border-radius:7px}
  .row{display:flex;gap:9px;flex-wrap:wrap;align-items:center}

  input[type=text],input:not([type]),select{font:inherit;border-radius:9px;
        border:1px solid var(--line);background:var(--bg);color:var(--txt);
        padding:9px 11px}
  select{cursor:pointer}
  .filebtn{background:var(--surface2);color:var(--txt);border-radius:9px;
           padding:9px 14px;cursor:pointer;font-weight:600;font-size:13px}
  .filebtn input{display:none}

  /* --- Галерея образцов --- */
  .samples{display:grid;grid-template-columns:repeat(auto-fill,minmax(78px,1fr));
           gap:9px;margin-bottom:14px}
  .samp{border:2px solid transparent;border-radius:11px;cursor:pointer;
        background:var(--bg);padding:4px;transition:.15s}
  .samp:hover{border-color:var(--line)}
  .samp img{width:100%;height:66px;object-fit:cover;border-radius:7px;display:block}
  .samp div{font-size:10px;color:var(--muted);margin-top:4px;text-align:center;
            overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

  /* --- Таблицы --- */
  .cols{display:grid;grid-template-columns:1fr 1fr;gap:20px}
  @media(max-width:880px){.cols{grid-template-columns:1fr}}
  table{width:100%;border-collapse:collapse}
  th,td{text-align:left;padding:9px 8px;font-size:13px;vertical-align:middle}
  th{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;
     letter-spacing:.4px;border-bottom:1px solid var(--line)}
  td{border-bottom:1px solid rgba(255,255,255,.04)}
  tr.removed{opacity:.4}
  .tag{display:inline-block;padding:3px 9px;border-radius:20px;font-size:11px;
       font-weight:700}
  .tag.white{background:rgba(78,203,136,.16);color:var(--ok)}
  .tag.black{background:rgba(250,95,107,.16);color:var(--no)}
  .tag.neutral{background:rgba(137,147,168,.18);color:var(--muted)}

  .addrow{display:flex;gap:9px;flex-wrap:wrap;margin-bottom:14px}
  .addrow input[type=text],.addrow input:not([type]){flex:1;min-width:120px}
  .err{color:var(--no);font-size:12px;margin-top:4px;min-height:15px}
  .muted{color:var(--muted)}

  /* --- Уведомление --- */
  #toast{position:fixed;left:50%;bottom:26px;transform:translateX(-50%) translateY(20px);
         background:var(--accent);color:#fff;padding:11px 20px;border-radius:10px;
         font-weight:600;font-size:13px;opacity:0;pointer-events:none;transition:.25s}
  #toast.show{opacity:1;transform:translateX(-50%) translateY(0)}

  /* --- Модальное окно добавления лица --- */
  .modal{position:fixed;inset:0;background:rgba(10,13,20,.74);z-index:50;
         display:none;align-items:center;justify-content:center}
  .modal.show{display:flex}
  .modal-box{background:var(--surface);border-radius:16px;padding:22px;
             width:340px;max-width:92vw;display:flex;flex-direction:column;gap:12px}
  .modal-h{font-size:16px;font-weight:700}
  .modal-prev{width:104px;height:104px;object-fit:cover;border-radius:12px;
              align-self:center;background:var(--bg);border:1px solid var(--line)}
  .modal-box input,.modal-box select{width:100%}
  .modal-act{display:flex;gap:9px}
  .modal-act button{flex:1}
  .camtip{font-size:12px;color:var(--muted)}
</style>
</head>
<body>

<header>
  <div class="brand">
    <div class="logo">SV</div>
    <div>
      <div class="title">SMART VISION</div>
      <div class="sub">распознавание лиц «свой / чужой»</div>
    </div>
  </div>
  <div class="controls">
    <div class="seg" id="modeSeg">
      <button data-mode="white_list" onclick="setMode('white_list')">Белый список</button>
      <button data-mode="black_list" onclick="setMode('black_list')">Чёрный список</button>
    </div>
    <div class="thr">
      <span>порог</span>
      <input type="range" id="thr" min="20" max="90" value="40"
             oninput="thrInput()" onchange="thrSave()">
      <span class="thrval" id="thrVal">40%</span>
    </div>
  </div>
</header>

<div class="modehint" id="modeHint"></div>

<!-- КАМЕРА -->
<div class="card">
  <div class="card-h">Живая камера</div>
  <div class="camera">
    <div class="camwrap" id="camWrap">
      <video id="cam" autoplay playsinline muted></video>
      <canvas id="overlay"></canvas>
      <div class="camhint" id="camHint">
        <div class="big">●</div>
        <div>нажмите «Включить камеру»</div>
      </div>
    </div>
    <div class="campanel">
      <div class="verdict idle" id="verdict">
        <div class="v-title" id="vTitle">Ожидание</div>
        <div class="v-reason" id="vReason">включите камеру или выберите фото ниже</div>
        <div class="v-bar"><div id="vBar"></div></div>
        <div class="v-meta" id="vMeta"></div>
      </div>
      <div class="row">
        <button id="camBtn" onclick="toggleCam()">Включить камеру</button>
        <button class="ghost" id="camSnap" onclick="camSnapshot()" disabled>
          Снимок в журнал</button>
      </div>
      <div class="camtip">Подсказка: кликните по лицу в кадре, чтобы
        добавить его в базу.</div>
      <div class="err" id="camErr"></div>
    </div>
  </div>
</div>

<!-- МОДАЛЬНОЕ ОКНО: ДОБАВЛЕНИЕ ЛИЦА -->
<div class="modal" id="modal">
  <div class="modal-box">
    <div class="modal-h">Добавить лицо в базу</div>
    <img class="modal-prev" id="mPreview" alt="лицо">
    <input id="mName" placeholder="Введите имя"
           onkeydown="if(event.key==='Enter')modalConfirm();
                      if(event.key==='Escape')closeModal()">
    <select id="mGroup"></select>
    <div class="modal-act">
      <button onclick="modalConfirm()">Добавить</button>
      <button class="ghost" onclick="closeModal()">Отмена</button>
    </div>
    <div class="err" id="mErr"></div>
  </div>
</div>

<!-- ПРОВЕРКА ФОТО -->
<div class="card">
  <div class="card-h">Проверка фото <b style="font-weight:400;text-transform:none;
       letter-spacing:0">— кликните по лицу или загрузите файл</b></div>
  <div class="samples" id="samples"></div>
  <div class="row">
    <label class="filebtn">Загрузить фото
      <input type="file" id="checkFile" accept="image/*" onchange="runCheckFile()"></label>
    <span class="err" id="checkErr" style="margin:0"></span>
  </div>
</div>

<!-- БАЗА ЛИЦ + ЖУРНАЛ -->
<div class="cols">
  <div class="card">
    <div class="card-h">База лиц</div>
    <div class="addrow">
      <input id="enrollName" placeholder="Имя нового лица">
      <select id="enrollGroup"></select>
      <button class="sm" onclick="enrollFromCamera()">С камеры</button>
      <label class="filebtn" style="padding:7px 12px;font-size:12px">Из файла
        <input type="file" id="enrollFile" accept="image/*" onchange="enrollFromFile()"></label>
    </div>
    <div class="err" id="enrollErr"></div>
    <input id="personSearch" placeholder="Поиск по имени..."
           oninput="renderPersons()" style="width:100%;margin-bottom:10px">
    <table>
      <thead><tr><th>Имя</th><th>Группа</th><th></th></tr></thead>
      <tbody id="personsBody"></tbody>
    </table>
  </div>

  <div class="card">
    <div class="card-h">Журнал проверок</div>
    <table>
      <thead><tr><th>Время</th><th>Лицо</th><th>%</th>
        <th>Решение</th><th>Причина</th></tr></thead>
      <tbody id="logBody"></tbody>
    </table>
  </div>
</div>

<div id="toast"></div>

<script>
const $=id=>document.getElementById(id);
let GROUPS=[], PERSONS=[], currentMode='white_list',
    lastRecog=null, pendingCrop=null;

async function api(url,opt){
  const r=await fetch(url,opt); const j=await r.json();
  if(j.error) throw new Error(j.error);
  return j;
}
function tagClass(lt){return lt==='white'?'white':lt==='black'?'black':'neutral';}

function toast(msg){
  const t=$('toast'); t.textContent=msg; t.classList.add('show');
  clearTimeout(t._tm); t._tm=setTimeout(()=>t.classList.remove('show'),2200);
}

const MODE_HINT={
  white_list:'Режим «Белый список»: доступ разрешён только своим с совпадением выше порога. Чужие, неизвестные и чёрный список — отказ.',
  black_list:'Режим «Чёрный список»: блокируются только лица из чёрного списка. Остальные распознанные лица проходят.'
};

// --- Режим и порог --------------------------------------------------
async function setMode(m){
  currentMode=m;
  applyMode(m);
  try{ await api('/api/mode',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({mode:m})}); }catch(e){}
}
function applyMode(m){
  currentMode=m;
  document.querySelectorAll('#modeSeg button').forEach(b=>
    b.classList.toggle('active',b.dataset.mode===m));
  $('modeHint').textContent=MODE_HINT[m];
}
function thrInput(){ $('thrVal').textContent=$('thr').value+'%'; }
let thrTm=null;
function thrSave(){
  clearTimeout(thrTm);
  thrTm=setTimeout(async()=>{
    try{ await api('/api/threshold',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({value:$('thr').value/100})});
      toast('Порог сохранён: '+$('thr').value+'%');
    }catch(e){}
  },350);
}

// --- Состояние ------------------------------------------------------
async function loadState(){ render(await api('/api/state')); }

function render(s){
  applyMode(s.mode);
  $('thr').value=Math.round(s.threshold*100); thrInput();
  GROUPS=s.groups;
  $('enrollGroup').innerHTML=GROUPS.map(g=>`<option>${g}</option>`).join('');

  PERSONS=s.persons;
  renderPersons();

  $('logBody').innerHTML=s.checks.map(c=>{
    const ok=c.decision==='ALLOWED';
    return `<tr>
      <td class="muted">${c.time.slice(11)}</td>
      <td>${c.name||'неизвестный'}</td>
      <td>${c.score}%</td>
      <td><span class="tag ${ok?'white':'black'}">${ok?'РАЗРЕШЁН':'ЗАПРЕЩЁН'}</span></td>
      <td class="muted" style="font-size:12px">${c.reason}</td></tr>`;
  }).join('')||'<tr><td colspan=5 class="muted">Журнал пуст</td></tr>';
}

function renderPersons(){
  const q=($('personSearch').value||'').trim().toLowerCase();
  const list=q ? PERSONS.filter(p=>p.name.toLowerCase().includes(q)) : PERSONS;
  $('personsBody').innerHTML=list.map(p=>{
    const opts=GROUPS.map(g=>
      `<option ${g===p.group?'selected':''}>${g}</option>`).join('');
    const removed=p.status==='removed';
    return `<tr class="${removed?'removed':''}">
      <td>${p.name}</td>
      <td><span class="tag ${tagClass(p.list_type)}">${p.group}</span></td>
      <td><div class="row" style="justify-content:flex-end">
        <select class="sm" onchange="moveP(${p.id},this.value)">${opts}</select>
        <button class="sm ghost" onclick="toggleP(${p.id},'${p.status}')">
          ${removed?'Вернуть':'Исключить'}</button>
      </div></td></tr>`;
  }).join('')||`<tr><td colspan=3 class="muted">`
    +(q?'Ничего не найдено':'База пуста')+`</td></tr>`;
}

// --- Блок вердикта --------------------------------------------------
function showVerdict(r){
  const ok=r.decision==='ALLOWED';
  $('verdict').className='verdict '+(ok?'allowed':'denied');
  $('vTitle').textContent=ok?'Доступ разрешён':'Доступ запрещён';
  $('vReason').textContent=r.reason;
  $('vBar').style.width=r.score+'%';
  $('vBar').style.background=ok?'var(--ok)':'var(--no)';
  $('vMeta').textContent=`совпадение ${r.score}% · `
    +`лицо: ${r.matched_name||r.name||'не распознано'} · `
    +`группа: ${r.matched_group||r.group||'—'}`;
}
function idleVerdict(text){
  $('verdict').className='verdict idle';
  $('vTitle').textContent='Ожидание';
  $('vReason').textContent=text||'наведите лицо в кадр';
  $('vBar').style.width='0';
  $('vMeta').textContent='';
}

// --- Проверка фото --------------------------------------------------
let SAMPLES=[];
async function loadSamples(){
  SAMPLES=(await api('/api/samples')).samples;
  $('samples').innerHTML=SAMPLES.map((s,i)=>
    `<div class="samp" onclick="runCheckSample(${i})">
       <img src="/img?path=${encodeURIComponent(s.path)}">
       <div>${s.name}</div></div>`).join('');
}
async function runCheckSample(i){
  $('checkErr').textContent='';
  try{
    const j=await api('/api/check',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({sample_path:SAMPLES[i].path,mode:currentMode})});
    showVerdict(j.result); render(j.state);
    toast('Проверка записана в журнал');
  }catch(e){ $('checkErr').textContent=e.message; }
}
async function runCheckFile(){
  const f=$('checkFile');
  if(!f.files.length) return;
  $('checkErr').textContent='';
  const fd=new FormData(); fd.append('image',f.files[0]); fd.append('mode',currentMode);
  try{
    const j=await api('/api/check',{method:'POST',body:fd});
    showVerdict(j.result); render(j.state);
    toast('Проверка записана в журнал');
  }catch(e){ $('checkErr').textContent=e.message; }
  f.value='';
}

// --- Добавление лица ------------------------------------------------
async function enrollFromFile(){
  const f=$('enrollFile'); if(!f.files.length) return;
  const name=$('enrollName').value.trim();
  if(!name){ $('enrollErr').textContent='Сначала введите имя'; f.value=''; return; }
  const fd=new FormData();
  fd.append('image',f.files[0]); fd.append('name',name);
  fd.append('group',$('enrollGroup').value);
  await doEnroll({method:'POST',body:fd});
  f.value='';
}
async function enrollFromCamera(){
  const name=$('enrollName').value.trim();
  if(!name){ $('enrollErr').textContent='Сначала введите имя'; return; }
  const data=grabFrame(640);
  if(!data){ $('enrollErr').textContent='Камера выключена — включите её выше'; return; }
  await doEnroll({method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({image_b64:data,name:name,group:$('enrollGroup').value})});
}
async function doEnroll(opt){
  $('enrollErr').textContent='';
  try{
    const j=await api('/api/enroll',opt);
    render(j.state); $('enrollName').value='';
    toast('Добавлено: '+j.info.full_name+' → '+j.info.group);
  }catch(e){ $('enrollErr').textContent=e.message; }
}
async function moveP(id,group){
  render((await api('/api/move',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({id,group})})).state);
  toast('Группа изменена');
}
async function toggleP(id,status){
  const v=status==='active'?'removed':'active';
  render((await api('/api/status',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({id,value:v})})).state);
}

// --- Живая камера ---------------------------------------------------
let camStream=null, camTimer=null, camBusy=false;
const _proc=document.createElement('canvas');
const PROC_W=480;

async function toggleCam(){
  if(camStream){ stopCam(); return; }
  $('camErr').textContent='';
  try{
    camStream=await navigator.mediaDevices.getUserMedia(
      {video:{width:{ideal:640},height:{ideal:480}}});
    const v=$('cam'); v.srcObject=camStream; await v.play();
    $('camHint').style.display='none';
    $('camWrap').classList.add('live');
    $('camBtn').textContent='Выключить камеру';
    $('camSnap').disabled=false;
    idleVerdict('наведите лицо в кадр');
    camTimer=setInterval(camTick,300);
  }catch(e){ $('camErr').textContent='Нет доступа к камере: '+e.message; }
}
function stopCam(){
  clearInterval(camTimer); camTimer=null;
  if(camStream){ camStream.getTracks().forEach(t=>t.stop()); camStream=null; }
  $('cam').srcObject=null; lastRecog=null;
  const o=$('overlay'); o.getContext('2d').clearRect(0,0,o.width,o.height);
  $('camHint').style.display='flex';
  $('camWrap').classList.remove('live');
  $('camBtn').textContent='Включить камеру';
  $('camSnap').disabled=true;
  idleVerdict('включите камеру или выберите фото ниже');
}
function grabFrame(width){
  const v=$('cam'), vw=v.videoWidth, vh=v.videoHeight;
  if(!vw) return null;
  const h=Math.round(width*vh/vw);
  _proc.width=width; _proc.height=h;
  _proc.getContext('2d').drawImage(v,0,0,width,h);
  return _proc.toDataURL('image/jpeg',0.7);
}
async function camTick(){
  if(camBusy||!camStream) return;
  const data=grabFrame(PROC_W);
  if(!data) return;
  camBusy=true;
  try{
    const j=await api('/api/recognize',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({image_b64:data,mode:currentMode})});
    drawFaces(j);
  }catch(e){}finally{ camBusy=false; }
}
function drawFaces(j){
  lastRecog=j;
  const o=$('overlay'); o.width=j.w; o.height=j.h;
  const ctx=o.getContext('2d'); ctx.clearRect(0,0,j.w,j.h);
  const fs=Math.round(j.w/20);
  ctx.font='bold '+fs+'px Segoe UI';
  for(const f of j.faces){
    const ok=f.decision==='ALLOWED', col=ok?'#4ecb88':'#fa5f6b';
    const [x,y,w,h]=f.box;
    ctx.lineWidth=Math.max(2,j.w/200); ctx.strokeStyle=col;
    ctx.strokeRect(x,y,w,h);
    drawLabel(ctx,f.name?`${f.name} · ${f.score}%`:'Неизвестный',x,y-7,col);
    drawLabel(ctx,ok?'Доступ разрешён':'Доступ запрещён',x,y+h+fs,col);
  }
  // вердикт по самому крупному лицу
  if(j.faces.length){
    let big=j.faces[0];
    for(const f of j.faces) if(f.box[2]*f.box[3]>big.box[2]*big.box[3]) big=f;
    showVerdict(big);
  }else if(camStream){
    idleVerdict('лицо не обнаружено');
  }
}
function drawLabel(ctx,text,x,y,col){
  ctx.lineWidth=4; ctx.strokeStyle='rgba(8,10,16,.85)';
  ctx.strokeText(text,x,y);
  ctx.fillStyle=col; ctx.fillText(text,x,y);
}
async function camSnapshot(){
  const data=grabFrame(640);
  if(!data){ $('camErr').textContent='Камера не готова'; return; }
  $('camErr').textContent='';
  try{
    const j=await api('/api/check',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({image_b64:data,mode:currentMode})});
    showVerdict(j.result); render(j.state);
    toast('Снимок записан в журнал');
  }catch(e){ $('camErr').textContent=e.message; }
}
window.addEventListener('beforeunload',()=>{ if(camStream) stopCam(); });

// --- Добавление лица кликом по кадру --------------------------------
function overlayClick(e){
  if(!camStream||!lastRecog) return;
  const o=$('overlay'), r=o.getBoundingClientRect();
  const cx=(e.clientX-r.left)*(o.width/r.width);
  const cy=(e.clientY-r.top)*(o.height/r.height);
  for(const f of lastRecog.faces){
    const [x,y,w,h]=f.box;
    if(cx>=x&&cx<=x+w&&cy>=y&&cy<=y+h){ openModal(f); return; }
  }
  toast('Кликните точно по рамке лица');
}
function cropFace(f){
  // Вырезаем лицо из кадра с запасом — сервер найдёт на нём одно лицо
  const v=$('cam'), sx=v.videoWidth/lastRecog.w;
  let [x,y,w,h]=f.box.map(n=>n*sx);
  const pad=0.45;
  x-=w*pad; y-=h*pad; w*=(1+2*pad); h*=(1+2*pad);
  x=Math.max(0,x); y=Math.max(0,y);
  w=Math.min(v.videoWidth-x,w); h=Math.min(v.videoHeight-y,h);
  const c=document.createElement('canvas'); c.width=w; c.height=h;
  c.getContext('2d').drawImage(v,x,y,w,h,0,0,w,h);
  return c.toDataURL('image/jpeg',0.85);
}
function openModal(f){
  pendingCrop=cropFace(f);
  $('mPreview').src=pendingCrop;
  $('mName').value=f.name||'';
  $('mGroup').innerHTML=GROUPS.map(g=>
    `<option ${g==='Свой'?'selected':''}>${g}</option>`).join('');
  $('mErr').textContent='';
  $('modal').classList.add('show');
  setTimeout(()=>$('mName').focus(),50);
}
function closeModal(){ $('modal').classList.remove('show'); pendingCrop=null; }
async function modalConfirm(){
  const name=$('mName').value.trim();
  if(!name){ $('mErr').textContent='Введите имя'; return; }
  if(!pendingCrop){ closeModal(); return; }
  $('mErr').textContent='';
  try{
    const j=await api('/api/enroll',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({image_b64:pendingCrop,name:name,
                           group:$('mGroup').value})});
    render(j.state); closeModal();
    toast('Добавлено: '+j.info.full_name+' → '+j.info.group);
  }catch(e){ $('mErr').textContent=e.message; }
}
$('overlay').addEventListener('click',overlayClick);
$('modal').addEventListener('click',e=>{ if(e.target===$('modal')) closeModal(); });

loadSamples();
loadState();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    run()
