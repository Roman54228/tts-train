Набор скриптов для скачивания данных с HuggingFace и обучения моделей [Piper TTS](https://github.com/OHF-Voice/piper1-gpl).

## Структура проекта


├── download.py         # Скачивание и подготовка датасета

├── train.py            # Запуск обучения из YAML-конфига

└── train_config.yaml   # Конфигурация обучения


## Использование

```
pip install -r req.txt
```

### 1. Скачивание и подготовка датасета

Скрипт скачивает датасет с HuggingFace, конвертирует MP3 в WAV (22050 Hz, mono), фильтрует по длительности и качеству, формирует metadata.csv в формате LJSpeech.

```
python download.py --dataset ESpeech/ESpeech-buldjat --output /media/4TB/ --tar "/home/ladmin/.cache/huggingface/hub/datasets--ESpeech--ESpeech-buldjat/snapshots/38a8fafff54069010e00c342e66838c7b8f9d105/buldjat_stripped_archive.tar"
 
```

Параметры:

| Флаг | Описание | По умолчанию |
|------|----------|--------------|
| --dataset | Название датасета на HuggingFace | (обязательный) |
| --output | Папка для сохранения данных | (обязательный) |
| --max-samples | Максимум аудио для обучения | 1500 |
| --voice-name | Имя голоса | из названия датасета |
| --tar | Путь до tar-архива (вместо скачивания) | — |

Результат:


<output>/buldjat_tts/
├── wavs/
│   ├── buldjat_00000.wav
│   └── ...
├── metadata.csv
└── config.json


### 2. Настройка конфига

Отредактируйте train_config.yaml под свои данные:


data:
  voice_name: "buldjat"
  csv_path: /path/to/metadata.csv
  audio_dir: /path/to/wavs
  cache_dir: /path/to/cache
  config_path: /path/to/config.json
  espeak_voice: "ru"
  batch_size: 30

model:
  sample_rate: 22050

trainer:
  accelerator: gpu
  devices: "[0]"
  precision: 16
  max_epochs: 5000


### 3. Запуск обучения


# Обучение с конфигом по умолчанию (train_config.yaml)
python train.py

# Свой конфиг
python train.py --config my_config.yaml

# Только показать команду без запуска
python train.py --dry-run


Скрипт train.py преобразует YAML-конфиг в аргументы командной строки и вызывает python3 -m piper.train fit.

## Фильтрация данных

При подготовке датасета применяются фильтры:

- Минимальная длина текста: 10 символов
- Длительность аудио: 0.5–15 секунд
- Минимальная амплитуда: 0.01 (отсев тишины)
- Нормализация громкости по пиковому значению