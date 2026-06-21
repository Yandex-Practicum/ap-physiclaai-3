"""Скачивание предсобранных артефактов (датасеты, чекпоинты) для CPU-потока.

Запуск:
    python3 scripts/download_artifacts.py --datasets train_1k train_10k eval
    python3 scripts/download_artifacts.py --checkpoints bc_1k
    python3 scripts/download_artifacts.py --checkpoints bc_1k bc_10k
    python3 scripts/download_artifacts.py --datasets train_1k train_10k eval --checkpoints bc_1k bc_10k
"""

import argparse
import os
import sys
import urllib.request
import zipfile


ARTIFACTS_BASE_URL = os.environ.get(
    "ARTIFACTS_URL",
    "https://storage.googleapis.com/robotics-course-practice3/artifacts"
)

DATASET_NAMES = {"train_1k", "train_10k", "eval"}
CHECKPOINT_NAMES = {"bc_1k", "bc_10k"}


def download_file(url: str, dest: str):
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"  Скачиваю: {url}")
    print(f"  → {dest}")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
        print()
    except Exception as e:
        print(f"\n  Ошибка скачивания: {e}")
        print("  Проверьте подключение к интернету и URL.")
        print(f"  Если URL недоступен, попросите преподавателя предоставить файл: {os.path.basename(dest)}")
        return False
    return True


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / 1024 / 1024
        total_mb = total_size / 1024 / 1024
        sys.stdout.write(f"\r  [{pct:3d}%] {mb:.1f} / {total_mb:.1f} MB")
    else:
        mb = downloaded / 1024 / 1024
        sys.stdout.write(f"\r  {mb:.1f} MB downloaded")
    sys.stdout.flush()


def download_dataset(name: str):
    url = f"{ARTIFACTS_BASE_URL}/datasets/{name}.zip"
    dest_zip = f"/tmp/{name}.zip"
    dest_dir = os.path.join("dataset", name)

    if os.path.exists(dest_dir) and len(os.listdir(dest_dir)) > 0:
        count = len([f for f in os.listdir(dest_dir) if f.endswith(".npz")])
        print(f"  Датасет {name} уже существует ({count} файлов). Пропуск.")
        return

    if download_file(url, dest_zip):
        os.makedirs(dest_dir, exist_ok=True)
        print(f"  Распаковка в {dest_dir}...")
        with zipfile.ZipFile(dest_zip, "r") as zf:
            zf.extractall(dest_dir)
        os.remove(dest_zip)
        count = len([f for f in os.listdir(dest_dir) if f.endswith(".npz")])
        print(f"  Готово: {count} файлов в {dest_dir}/")


def download_checkpoint(name: str):
    ckpt_dir = os.path.join("logs", name, "checkpoints")
    log_dir = os.path.join("logs", name)

    for fname in ["best.pt", "last.pt"]:
        dest = os.path.join(ckpt_dir, fname)
        if os.path.exists(dest):
            print(f"  Чекпоинт {name}/{fname} уже существует. Пропуск.")
            continue
        url = f"{ARTIFACTS_BASE_URL}/checkpoints/{name}/{fname}"
        download_file(url, dest)

    tb_url = f"{ARTIFACTS_BASE_URL}/checkpoints/{name}/tb_logs.zip"
    tb_zip = f"/tmp/{name}_tb.zip"
    if download_file(tb_url, tb_zip):
        print(f"  Распаковка TensorBoard-логов в {log_dir}...")
        with zipfile.ZipFile(tb_zip, "r") as zf:
            zf.extractall(log_dir)
        os.remove(tb_zip)
        print(f"  Готово: TensorBoard-логи в {log_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Скачивание артефактов Practice_3")
    parser.add_argument("--datasets", nargs="*", default=[],
                        help="Датасеты для скачивания: train_1k, train_10k, eval")
    parser.add_argument("--checkpoints", nargs="*", default=[],
                        help="Чекпоинты для скачивания: bc_1k, bc_10k")
    args = parser.parse_args()

    if not args.datasets and not args.checkpoints:
        print("Укажите --datasets и/или --checkpoints. Пример:")
        print("  python3 scripts/download_artifacts.py --datasets train_1k train_10k eval")
        print("  python3 scripts/download_artifacts.py --checkpoints bc_1k bc_10k")
        return

    for name in args.datasets:
        if name not in DATASET_NAMES:
            print(f"Неизвестный датасет: {name}. Допустимые: {DATASET_NAMES}")
            continue
        print(f"\n=== Датасет: {name} ===")
        download_dataset(name)

    for name in args.checkpoints:
        if name not in CHECKPOINT_NAMES:
            print(f"Неизвестный чекпоинт: {name}. Допустимые: {CHECKPOINT_NAMES}")
            continue
        print(f"\n=== Чекпоинт: {name} ===")
        download_checkpoint(name)

    print("\nВсе артефакты обработаны.")


if __name__ == "__main__":
    main()
