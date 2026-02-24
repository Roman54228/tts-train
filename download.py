#!/usr/bin/env python3
"""
Скачивание датасета с HuggingFace и подготовка в формате LJSpeech для Piper TTS.

Пример использования:
    python download.py --dataset ESpeech/ESpeech-buldjat --output /home/ladmin/e_data
    python download.py --dataset ESpeech/ESpeech-buldjat --output /home/ladmin/e_data --tar /path/to/data.tar

Результат:
    <output>/buldjat_tts/
    ├── wavs/
    │   ├── buldjat_00000.wav
    │   └── ...
    ├── metadata.csv
    └── config.json

Затем запуск обучения:
    python3 -m piper.train fit \\
        --data.voice_name "buldjat" \\
        --data.csv_path <output>/buldjat_tts/metadata.csv \\
        --data.audio_dir <output>/buldjat_tts/wavs \\
        ...
"""

import argparse
import json
import os
import sys
import tarfile

import librosa
import numpy as np
import soundfile as sf
from datasets import load_dataset


SAMPLE_RATE = 22050
MIN_TEXT_LEN = 10
MIN_DURATION = 0.5
MAX_DURATION = 15.0
MIN_PEAK = 0.01


def parse_args():
    parser = argparse.ArgumentParser(
        description="Скачать датасет с HuggingFace и подготовить для Piper TTS"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Название датасета на HuggingFace (например: ESpeech/ESpeech-buldjat)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Корневая папка для сохранения данных (например: /home/ladmin/e_data)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1500,
        help="Максимальное количество сэмплов (по умолчанию: 1500)",
    )
    parser.add_argument(
        "--voice-name",
        default=None,
        help="Имя голоса (по умолчанию: берётся из названия датасета)",
    )
    parser.add_argument(
        "--tar",
        default=None,
        help="Путь до tar-архива с датасетом (если не указан — ищется в кэше HuggingFace)",
    )
    return parser.parse_args()


def download_dataset(dataset_name: str) -> tuple:
    """Скачивает датасет и возвращает (dataset, путь к tar-архиву)."""
    print(f"Скачиваем {dataset_name}...")
    ds = load_dataset(dataset_name, split="train")
    print(f"Датасет загружен: {len(ds)} сэмплов")

    # Ищем tar-архив в кэше
    from datasets import config as ds_config

    cache_base = os.path.join(
        str(ds_config.HF_DATASETS_CACHE),
        "hub",
        f"datasets--{dataset_name.replace('/', '--')}",
    )

    tar_path = None
    if os.path.isdir(cache_base):
        for root, _, files in os.walk(cache_base):
            for f in files:
                if f.endswith(".tar"):
                    tar_path = os.path.join(root, f)
                    break
            if tar_path:
                break

    return ds, tar_path


def extract_archive(tar_path: str, extract_dir: str):
    """Распаковывает tar-архив."""
    print(f"Распаковка {tar_path}...")
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(extract_dir)
    print(f"Распаковано в {extract_dir}/")


def prepare_dataset(extract_dir: str, out_dir: str, voice_name: str, max_samples: int):
    """Конвертирует MP3 в WAV и формирует metadata.csv."""
    wavs_dir = os.path.join(out_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    sample_count = 0
    metadata_lines = []
    skipped = 0

    for root, _, files in os.walk(extract_dir):
        json_files = [f for f in files if f.endswith(".json")]
        if not json_files:
            continue

        json_path = os.path.join(root, json_files[0])
        with open(json_path, "r", encoding="utf-8") as f:
            segments = json.load(f)

        audio_files = [f for f in files if f.endswith(".mp3")]
        audio_files.sort(key=lambda x: int(x.replace(".mp3", "").split("_")[-1]))

        if len(audio_files) != len(segments):
            folder_id = os.path.basename(root)
            print(
                f"  WARN: {folder_id} — {len(audio_files)} аудио vs "
                f"{len(segments)} сегментов, пропускаю"
            )
            skipped += len(audio_files)
            continue

        for audio_file, segment in zip(audio_files, segments):
            if sample_count >= max_samples:
                break

            text = segment["text"].strip()
            if len(text) < MIN_TEXT_LEN:
                skipped += 1
                continue

            mp3_path = os.path.join(root, audio_file)
            try:
                y, _ = librosa.load(mp3_path, sr=SAMPLE_RATE, mono=True)
            except Exception:
                skipped += 1
                continue

            duration = len(y) / SAMPLE_RATE
            if duration < MIN_DURATION or duration > MAX_DURATION:
                skipped += 1
                continue

            peak = np.max(np.abs(y))
            if peak < MIN_PEAK:
                skipped += 1
                continue
            y = y / peak

            wav_name = f"{voice_name}_{sample_count:05d}.wav"
            sf.write(os.path.join(wavs_dir, wav_name), y, SAMPLE_RATE)
            metadata_lines.append(f"{wav_name}|{text}")
            sample_count += 1

        if sample_count >= max_samples:
            break

    # metadata.csv
    csv_path = os.path.join(out_dir, "metadata.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata_lines))

    print(f"\nГотово: {sample_count} сэмплов, пропущено: {skipped}")
    return sample_count


def print_summary(out_dir: str, voice_name: str):
    """Выводит итоговую информацию."""
    wavs_dir = os.path.join(out_dir, "wavs")
    csv_path = os.path.join(out_dir, "metadata.csv")

    wav_count = len(os.listdir(wavs_dir)) if os.path.isdir(wavs_dir) else 0
    print(f"\nWAV файлы: {wav_count}")

    if os.path.isfile(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"Строк в metadata.csv: {len(lines)}")
        print("Первые 3 записи:")
        for line in lines[:3]:
            print(f"  {line.strip()}")

    print(f"\nДатасет готов: {out_dir}/")
    print(f"\nПример команды обучения:")
    print(f"  python3 -m piper.train fit \\")
    print(f'    --data.voice_name "{voice_name}" \\')
    print(f"    --data.csv_path {csv_path} \\")
    print(f"    --data.audio_dir {wavs_dir} \\")
    print(f"    --model.sample_rate {SAMPLE_RATE} \\")
    print(f'    --data.espeak_voice "ru" \\')
    print(f"    --data.cache_dir {out_dir}/cache \\")
    print(f"    --data.config_path {out_dir}/config.json")


def main():
    args = parse_args()

    # Определяем имя голоса из датасета, если не указано
    voice_name = args.voice_name
    if not voice_name:
        # ESpeech/ESpeech-buldjat -> buldjat
        voice_name = args.dataset.split("/")[-1].split("-")[-1]

    out_dir = os.path.join(args.output, f"{voice_name}_tts")

    # 1. Скачиваем / определяем tar
    if args.tar:
        tar_path = args.tar
        if not os.path.isfile(tar_path):
            print(f"ОШИБКА: файл не найден: {tar_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Используем указанный tar-архив: {tar_path}")
    else:
        _, tar_path = download_dataset(args.dataset)
        if not tar_path:
            print("ОШИБКА: tar-архив не найден в кэше датасета", file=sys.stderr)
            sys.exit(1)

    # 2. Распаковываем
    extract_dir = os.path.join(args.output, f"{voice_name}_raw")
    extract_archive(tar_path, extract_dir)

    # 3. Конвертируем
    print(f"\nПодготовка датасета (макс. {args.max_samples} сэмплов)...")
    count = prepare_dataset(extract_dir, out_dir, voice_name, args.max_samples)

    if count == 0:
        print("ОШИБКА: не удалось подготовить ни одного сэмпла", file=sys.stderr)
        sys.exit(1)

    # 4. Итог
    print_summary(out_dir, voice_name)


if __name__ == "__main__":
    main()
