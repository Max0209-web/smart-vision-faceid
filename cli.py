"""Командный интерфейс системы распознавания лиц "свой-чужой".

Примеры:
    python cli.py init
    python cli.py seed
    python cli.py enroll "Иван Иванов" path/to/face.jpg
    python cli.py enroll "Злодей" path/to/face.jpg --group "Чёрный список"
    python cli.py list
    python cli.py remove 3
    python cli.py move 3 --group "Чёрный список"
    python cli.py check path/to/test.jpg --mode white_list
    python cli.py mode white_list
    python cli.py threshold 0.4
    python cli.py log
    python cli.py serve
"""
import argparse
import sys

import config


def banner(text: str) -> None:
    print("=" * 60)
    print(text)
    print("=" * 60)


def cmd_init(args, svc):
    banner("Инициализация базы данных")
    print(f"База готова: {config.DB_PATH}")
    print("Группы:")
    import database as db
    for g in db.list_groups():
        print(f"  #{g['id']} {g['name']} [{g['list_type']}]")


def cmd_enroll(args, svc):
    info = svc.enroll(args.name, args.image, args.group)
    banner("Лицо добавлено в базу")
    print(f"  ID:     {info['person_id']}")
    print(f"  Имя:    {info['full_name']}")
    print(f"  Группа: {info['group']}")


def cmd_list(args, svc):
    banner("Лица в базе")
    rows = svc.list_persons()
    if not rows:
        print("  (пусто)")
        return
    print(f"  {'ID':<4}{'Имя':<26}{'Группа':<18}{'Статус':<10}{'Призн.'}")
    print("  " + "-" * 62)
    for r in rows:
        print(f"  {r['id']:<4}{r['full_name']:<26}{r['group_name']:<18}"
              f"{r['status']:<10}{r['features']}")


def cmd_remove(args, svc):
    svc.remove_from_own(args.id)
    p = __import__("database").get_person(args.id)
    banner("Лицо исключено из группы 'Свой'")
    print(f"  #{args.id} {p['full_name']} -> группа '{p['group_name']}'")


def cmd_move(args, svc):
    svc.move_to_group(args.id, args.group)
    p = __import__("database").get_person(args.id)
    banner("Группа изменена")
    print(f"  #{args.id} {p['full_name']} -> '{p['group_name']}'")


def cmd_status(args, svc):
    svc.set_status(args.id, args.value)
    banner("Статус изменён")
    print(f"  #{args.id} -> {args.value}")


def cmd_check(args, svc):
    res = svc.check(args.image, args.mode)
    banner("Результат проверки")
    print(f"  Файл:        {res['image_path']}")
    print(f"  Режим:       {res['mode']}")
    print(f"  Совпадение:  {res['score']}%")
    print(f"  Лицо в БД:   {res['matched_name'] or 'не распознано'}")
    print(f"  Группа:      {res['matched_group'] or '-'}")
    mark = "[+] ДОСТУП РАЗРЕШЁН" if res["decision"] == "ALLOWED" else "[x] ДОСТУП ЗАПРЕЩЁН"
    print(f"  Решение:     {mark}")
    print(f"  Причина:     {res['reason']}")


def cmd_mode(args, svc):
    svc.set_mode(args.value)
    print(f"Режим установлен: {args.value}")


def cmd_threshold(args, svc):
    svc.set_threshold(args.value)
    print(f"Порог установлен: {args.value}")


def cmd_log(args, svc):
    banner(f"Журнал проверок (последние {args.limit})")
    rows = svc.list_checks(args.limit)
    if not rows:
        print("  (журнал пуст)")
        return
    for r in rows:
        mark = "РАЗРЕШЁН" if r["decision"] == "ALLOWED" else "ЗАПРЕЩЁН"
        print(f"  [{r['checked_at']}] {mark} | {r['score']}% | "
              f"{r['matched_name'] or 'unknown'} | {r['reason']}")


def cmd_seed(args, svc):
    import seed
    seed.run(svc)


def cmd_serve(args, svc):
    import webapp
    webapp.run(args.host, args.port, args.cert, args.key)


def cmd_camera(args, svc):
    import camera
    camera.run(args.index, args.mode)


def build_parser():
    p = argparse.ArgumentParser(
        description="Система распознавания лиц 'свой-чужой'",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="создать/проверить базу данных")

    e = sub.add_parser("enroll", help="добавить лицо в базу")
    e.add_argument("name", help="имя человека")
    e.add_argument("image", help="путь к изображению лица")
    e.add_argument("--group", default=config.GROUP_OWN,
                   help=f"группа (по умолчанию '{config.GROUP_OWN}')")

    sub.add_parser("list", help="показать все лица")

    r = sub.add_parser("remove", help="исключить лицо из группы 'Свой'")
    r.add_argument("id", type=int)

    m = sub.add_parser("move", help="перевести лицо в другую группу")
    m.add_argument("id", type=int)
    m.add_argument("--group", required=True)

    st = sub.add_parser("status", help="изменить статус лица (active/removed)")
    st.add_argument("id", type=int)
    st.add_argument("value", choices=["active", "removed"])

    c = sub.add_parser("check", help="проверить тестовое изображение")
    c.add_argument("image")
    c.add_argument("--mode", choices=["white_list", "black_list"], default=None)

    md = sub.add_parser("mode", help="режим по умолчанию")
    md.add_argument("value", choices=["white_list", "black_list"])

    th = sub.add_parser("threshold", help="порог совпадения (0..1)")
    th.add_argument("value", type=float)

    lg = sub.add_parser("log", help="журнал проверок")
    lg.add_argument("--limit", type=int, default=20)

    sub.add_parser("seed", help="загрузить демо-набор лиц из faces_data")

    sv = sub.add_parser("serve", help="запустить веб-интерфейс")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--cert", default=None, help="путь к TLS-сертификату (HTTPS)")
    sv.add_argument("--key", default=None, help="путь к ключу TLS (HTTPS)")

    cam = sub.add_parser("camera", help="режим живой камеры (окно OpenCV)")
    cam.add_argument("--index", type=int, default=0, help="номер камеры")
    cam.add_argument("--mode", choices=["white_list", "black_list"], default=None)

    return p


HANDLERS = {
    "init": cmd_init, "enroll": cmd_enroll, "list": cmd_list,
    "remove": cmd_remove, "move": cmd_move, "status": cmd_status,
    "check": cmd_check, "mode": cmd_mode, "threshold": cmd_threshold,
    "log": cmd_log, "seed": cmd_seed, "serve": cmd_serve,
    "camera": cmd_camera,
}


def main():
    args = build_parser().parse_args()
    from service import FaceService, ServiceError
    try:
        svc = FaceService()
        HANDLERS[args.command](args, svc)
    except (ServiceError, FileNotFoundError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
