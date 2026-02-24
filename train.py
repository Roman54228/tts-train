#!/usr/bin/env python3
"""
Запуск обучения Piper TTS из YAML-конфига.

Использование:
    python train.py                         # используется train_config.yaml
    python train.py --config my_config.yaml # свой файл конфига
"""

import argparse
import subprocess
import sys

import yaml


def main():
    parser = argparse.ArgumentParser(description="Запуск обучения Piper TTS из конфига")
    parser.add_argument(
        "--config",
        default="train_config.yaml",
        help="Путь к YAML-конфигу (по умолчанию: train_config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только показать команду, не запускать",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Собираем аргументы командной строки из конфига
    cli_args = ["python3", "-m", "piper.train", "fit"]

    for section, params in config.items():
        if not isinstance(params, dict):
            continue
        for key, value in params.items():
            if value is None:
                continue
            cli_args.append(f"--{section}.{key}")
            cli_args.append(str(value))

    print("Команда:")
    print(" \\\n  ".join(cli_args))
    print()

    if args.dry_run:
        print("(dry-run, обучение не запущено)")
        return

    print("Запуск обучения...\n")
    result = subprocess.run(cli_args)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
