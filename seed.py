"""Загрузка демонстрационного набора лиц из faces_data/face в SQL-базу.

Раскладка для готовой демонстрации трёх сценариев заказчика:
  * группа "Свой"          - сотрудники с доступом;
  * группа "Чёрный список" - заблокированные лица;
  * один человек не загружается - используется как "unknown / чужой".
"""
import os

import config

# Демо-распределение по группам (имена папок в faces_data/face)
OWN = [
    "aleksey_nalivayka", "alla_pukacheva", "danil_lubariskiy",
    "daria_hleb", "daria_mamonova", "kirill_gluharev",
    "kristina_sherman", "leonid_kaprashev", "maria_bolt",
]
BLACKLIST = ["gleb_garman", "kirill_sofhob"]
# "ulyana_salomonova" намеренно не загружается -> тест "неизвестное лицо"


def _pretty(folder: str) -> str:
    return folder.replace("_", " ").title()


def _find_image(folder: str):
    d = os.path.join(config.DATASET_DIR, folder)
    if not os.path.isdir(d):
        return None
    for f in sorted(os.listdir(d)):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
            return os.path.join(d, f)
    return None


def run(svc) -> None:
    if not os.path.isdir(config.DATASET_DIR):
        print(f"Папка с лицами не найдена: {config.DATASET_DIR}")
        return

    print("=" * 60)
    print("Загрузка демо-набора лиц")
    print("=" * 60)

    plan = [(name, config.GROUP_OWN) for name in OWN]
    plan += [(name, config.GROUP_BLACKLIST) for name in BLACKLIST]

    ok, skipped = 0, 0
    for folder, group in plan:
        img = _find_image(folder)
        if img is None:
            print(f"  [-] {folder}: изображение не найдено")
            skipped += 1
            continue
        try:
            info = svc.enroll(_pretty(folder), img, group)
            print(f"  [+] #{info['person_id']:<3} {info['full_name']:<24} -> {group}")
            ok += 1
        except Exception as exc:
            print(f"  [-] {folder}: {exc}")
            skipped += 1

    print("-" * 60)
    print(f"Загружено: {ok}, пропущено: {skipped}")
    print()
    print("Демо-сценарии:")
    print("  1. Разрешённый: проверьте лицо из группы 'Свой'")
    print("  2. Исключение:  remove <id> 'своего', затем повторная проверка")
    print("  3. Запрет:      проверьте лицо из чёрного списка")
    print("     или 'ulyana_salomonova' (не в базе) -> неизвестное лицо")


if __name__ == "__main__":
    from service import FaceService
    run(FaceService())
