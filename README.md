# avito_bot_detection

Тестовое задание на Avito DS Bootcamp: детекция ботов по потоку событий.
Условие задачи — в [TASK.md](TASK.md). Решение находится в [bot_detection_challenge/](bot_detection_challenge).

## Структура репозитория

```
.
├── TASK.md                     # условие задачи
├── dataset_2589118_6.txt       # исходное описание/ссылка на данные из задания
└── bot_detection_challenge/    # решение
    ├── solution.ipynb          # итоговый ноутбук: подход, код, обучение, submission.csv
    ├── quickstart.ipynb        # стартовый ноутбук с обзором данных и базовыми моделями
    ├── run_solution.py         # запуск solution.ipynb из чистого Jupyter-ядра (python run_solution.py)
    ├── metric.py                # официальная метрика: Precision @ Recall >= 0.70
    │
    ├── behavior_features.py     # расчет поведенческих признаков (ритм, разнообразие, повторяемость)
    ├── pointer_features.py      # расчет совместных XY-признаков (координаты кликов/позиций)
    ├── behavior_experiments.py  # эксперименты по отбору поведенческих признаков
    ├── pointer_experiments.py   # эксперименты по отбору XY-признаков
    ├── test_behavior_features.py
    ├── test_pointer_features.py
    │
    ├── selected_features.json   # итоговый набор из 30 поведенческих признаков
    ├── pointer_selection.json   # итоговый набор из 12 XY-признаков
    ├── BEHAVIOR_REPORT.md       # отчет по отбору поведенческих признаков
    ├── POINTER_REPORT.md        # отчет по эксперименту с XY-признаками
    │
    ├── requirements.txt         # зафиксированные версии библиотек для решения
    ├── requirements-notebook.txt# версии для запуска самого Jupyter-ядра
    │
    ├── data/                    # исходные данные задания
    │   ├── train.csv
    │   ├── test.csv
    │   └── events.csv.gz
    ├── sample_submission.csv    # пример формата ответа
    ├── submission.csv           # итоговый submission (результат solution.ipynb)
    │
    ├── artifacts/                # сгенерированные артефакты экспериментов (в .gitignore)
    └── submissions/               # промежуточные варианты submission (в .gitignore)
```

## Воспроизведение

1. Выбрать Python 3.14.3 с зависимостями из `bot_detection_challenge/requirements-notebook.txt`.
2. Открыть `bot_detection_challenge/solution.ipynb` → **Restart Kernel and Run All**.

Либо из корня репозитория:

```
python bot_detection_challenge/run_solution.py
```

Итоговая модель — CatBoost на 42 признаках (30 поведенческих + 12 совместных XY).
Результат воспроизведения — `bot_detection_challenge/submission.csv`, совпадающий с
сохраненным `submissions/catboost_behavior_xy.csv`.