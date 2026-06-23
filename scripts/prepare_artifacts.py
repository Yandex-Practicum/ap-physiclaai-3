"""Подготовка артефактов для облака (запускается при подготовке курса, не студентами).

Упаковывает локально собранные датасеты и обученные BC-чекпоинты в структуру,
которую ожидает scripts/download_artifacts.py:

    <ARTIFACTS_URL>/datasets/<name>.zip            -> распаковывается в dataset/<name>/
    <ARTIFACTS_URL>/checkpoints/<name>/best.pt
    <ARTIFACTS_URL>/checkpoints/<name>/last.pt
    <ARTIFACTS_URL>/checkpoints/<name>/tb_logs.zip  -> распаковывается в logs/<name>/

Складывает всё в локальную папку (по умолчанию ./artifacts_upload/) с той же
иерархией и печатает готовые команды gsutil для заливки в GCS-бакет.

Запуск:
    python3 scripts/prepare_artifacts.py \
        --datasets train_1k train_10k eval \
        --checkpoints bc_1k bc_10k \
        --gcs gs://<ваш-бакет>/artifacts
"""

import argparse
import glob
import os
import shutil
import zipfile


def zip_files(files, dest_zip):
    os.makedirs(os.path.dirname(dest_zip) or ".", exist_ok=True)
    with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=os.path.basename(f))  # файлы в корне архива
    return dest_zip


def prepare_dataset(name, out):
    src = os.path.join("dataset", name)
    files = sorted(glob.glob(os.path.join(src, "*.npz")))
    if not files:
        print(f"  [пропуск] нет .npz в {src}")
        return None
    dest = os.path.join(out, "datasets", f"{name}.zip")
    zip_files(files, dest)
    print(f"  {src} ({len(files)} файлов) -> {dest}")
    return dest


def prepare_checkpoint(name, out):
    ck = os.path.join("logs", name, "checkpoints")
    dst = os.path.join(out, "checkpoints", name)
    os.makedirs(dst, exist_ok=True)
    made = []
    for fn in ("best.pt", "last.pt"):
        src = os.path.join(ck, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst, fn))
            made.append(fn)
    # tb_logs.zip — только events-файлы (без подпапки checkpoints)
    events = sorted(glob.glob(os.path.join("logs", name, "events.*")))
    if events:
        zip_files(events, os.path.join(dst, "tb_logs.zip"))
        made.append("tb_logs.zip")
    if made:
        print(f"  logs/{name} -> {dst} ({', '.join(made)})")
    else:
        print(f"  [пропуск] нет чекпоинтов/логов в logs/{name}")
    return dst if made else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="*", default=[])
    p.add_argument("--checkpoints", nargs="*", default=[])
    p.add_argument("--out", default="artifacts_upload")
    p.add_argument("--gcs", default="", help="например, gs://my-bucket/artifacts")
    args = p.parse_args()

    if not args.datasets and not args.checkpoints:
        print("Укажите --datasets и/или --checkpoints. Пример:")
        print("  python3 scripts/prepare_artifacts.py --datasets train_1k eval --checkpoints bc_1k")
        return

    print(f"=== Упаковка в {args.out}/ ===")
    for name in args.datasets:
        prepare_dataset(name, args.out)
    for name in args.checkpoints:
        prepare_checkpoint(name, args.out)

    print("\n=== Готово. Заливка в облако ===")
    if args.gcs:
        print(f"  gsutil -m cp -r {args.out}/* {args.gcs}/")
        print(f"  gsutil -m acl ch -r -u AllUsers:R {args.gcs}/   # сделать публично читаемым (по необходимости)")
        print(f"\n  Затем у студентов: export ARTIFACTS_URL={args.gcs.replace('gs://', 'https://storage.googleapis.com/')}")
    else:
        print(f"  Залейте содержимое {args.out}/ в облако, сохранив структуру (datasets/, checkpoints/).")
        print("  Для GCS: повторите запуск с --gcs gs://<бакет>/artifacts, чтобы получить точные команды.")


if __name__ == "__main__":
    main()
