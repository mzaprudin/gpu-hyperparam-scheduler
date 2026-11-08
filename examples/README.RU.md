# Примеры запуска и входные данные

[English version](README.md) · [Обзор проекта](../README.RU.md) ·
[Архитектура](../ARCHITECTURE.RU.md)

Пример для CPU обучает классификатор временных рядов и сохраняет модель Keras,
которую можно загрузить повторно, JSON-отчёт и график распределения вероятностей.
Выполняйте команды из корня проекта: пути к данным и конфигурациям отсчитываются
от текущего рабочего каталога.

## Обучение на CPU

Используйте Python 3.12 и зафиксированные версии зависимостей:

```bash
uv sync --locked
uv run gpu-hyperparam-demo-data
uv run gpu-hyperparam-train --config examples/cpu.json
```

Генератор создаёт `data/demo.pkl`: три синтетических ряда по 720 строк,
по одному наблюдению в минуту, с seed `42`. Конфигурация CPU также задаёт
seed `42`; небольшая модель обучается одну эпоху на окнах длиной 16 строк.

Модель обучается на ранней части данных, выбирает порог уверенности на validation,
затем оценивает test с этим порогом. В консоли выводится сравнение метрик validation
и test. Поиск ранжирует конфигурации по precision из отчёта test.
Полный тракт описан в [архитектуре](../ARCHITECTURE.RU.md).

| Локальный файл | Содержимое |
| --- | --- |
| `artifacts/cpu-demo/demo.keras` | Обученная модель со своими слоями и функцией потерь |
| `artifacts/cpu-demo/demo.json` | Метрики test, параметры модели и обучения |
| `artifacts/cpu-demo/demo.png` | Распределение вероятностей и анализ порога |
| `data/demo_mg.pkl` | Подготовленные генераторы train, validation и test |

Исходные данные остаются в `data/demo.pkl`. Опубликованные
[отчёт CPU](results/cpu-demo.json) и [график вероятностей](../docs/assets/cpu-demo.png)
получены при завершённом запуске этого примера.

## Оценка сохранённой модели

Загрузите модель и оцените test с явно заданным порогом `0.5`:

```bash
uv run gpu-hyperparam-train --config examples/cpu.json \
  --test --model artifacts/cpu-demo/demo.keras \
  --threshold 0.5 --save-dir artifacts/cpu-evaluation
```

Команда запишет `demo.json` и `demo.png` в `artifacts/cpu-evaluation/`, сохранив
отчёт и график обучения. В режиме `--test` параметр `--threshold` напрямую задаёт
порог оценки. Для оценки с порогом, выбранным при обучении, возьмите значение
`results.optimal_threshold` из исходного JSON-отчёта.

Явные параметры CLI имеют приоритет над JSON. Ключи конфигурации используют
имена аргументов с подчёркиваниями: `save_dir`, `val_hours`, `min_epochs`.

## Собственные данные

Передайте `--data path/to/data.pkl`: доверенный pickle с содержимым
`dict[str, pandas.DataFrame]`, по одному хронологическому ряду на идентификатор.
В каждом DataFrame нужны следующие числовые столбцы:

| Столбец | Значение |
| --- | --- |
| `block_number` | Монотонно растущий номер наблюдения |
| `unix_timestamp` | Время наблюдения в секундах Unix |
| `transactions`, `accounts` | Показатели активности |
| `buy_volume`, `sell_volume` | Объёмы торговли |
| `price` | Положительная цена |
| `liquidity_token`, `liquidity_sol` | Признаки ликвидности |

Загружайте pickle только из доверенного источника: десериализация выполняет Python.
В каждой выборке должно хватать пригодных окон на пакет; в сбалансированной
обучающей выборке нужны оба класса. `--data` также принимает кеши генераторов,
созданные этим пакетом. Исторические кеши Neurosol пересоздайте из исходных
DataFrame, чтобы использовать нынешние пути пакета вместо прежних импортов `src.*`.

## Поиск на GPU и продолжение

Нужны Linux, совместимые GPU NVIDIA и установленный драйвер NVIDIA. Установите
необязательные зависимости CUDA, проверьте видимые устройства и запустите поиск:

```bash
uv sync --locked --extra gpu
uv run gpu-hyperparam-check
uv run gpu-hyperparam-demo-data
uv run gpu-hyperparam-scheduler --config examples/search.json --gpus 0,1
```

`--gpus` задаёт индексы в списке устройств, видимых TensorFlow. При
`CUDA_VISIBLE_DEVICES=2,3` параметры `--gpus 0,1` выбирают физические GPU 2 и 3.
На каждом GPU работает отдельный процесс, запущенный через `spawn`.

История сохраняется в `artifacts/search/optimization_results.json`. Продолжите
этот поиск с бюджетом до пяти дополнительных конфигураций:

```bash
uv run gpu-hyperparam-scheduler --config examples/search.json \
  --gpus 0,1 --max-configurations 5
```

При продолжении используйте те же данные и настройки; для каждого эксперимента
задавайте отдельный `--save-dir`. Параметры `--initial-points`, `--min-epochs`,
`--max-epochs` и `--genetic-only` управляют бюджетом и стратегией поиска.
