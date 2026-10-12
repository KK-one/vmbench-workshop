# VMBench workshop

Студенческий практикум по PAS, OIS, TCS, MSS и CAS на официальном VMBench.

- `vmbench_workshop_colab.ipynb` — объяснения, конфигурация, отдельные вызовы метрик и задания.
- `utils.py` — установка, lazy checkpoints, работа с видео, cache, CSV, графики и запуск evaluator’ов в отдельных процессах.

## Открыть в Colab

Загрузите notebook в https://colab.research.google.com/ или откройте его из этого repo. Выберите T4 GPU и выполните Runtime → Run all.

Первая ячейка подключает соседний `utils.py`. Если его нет в runtime, она скачивает файл по `UTILS_URL`. Для приватного repo или недоступного URL появится upload-диалог: выберите `utils.py`, затем в отдельном диалоге загрузите видео. SSH-ключ и токены студентам не нужны. После публикации repo открытый raw URL позволяет обойти upload модуля.

Для локального запуска kernel должен видеть папку с `utils.py`. Импорт модуля не устанавливает зависимости, не скачивает веса и не создаёт каталогов. Явная инициализация:

```python
from utils import WorkshopConfig, configure, compute_pas

workshop = configure(WorkshopConfig(work_dir="/content", gpu_available=True, gpu_gib=15))
score = compute_pas("/content/videos/example.mp4")
```

В notebook наличие и объём GPU определяются отдельной ячейкой, затем передаются в конфигурацию автоматически. Все функции читают состояние объекта `workshop`, а не переменные notebook. При повторном выполнении конфигурации в той же VM объект и результаты сохраняются. После reset нужно заново выполнить Run all.

## Официальный код и ограничения

Источник: [AMAP-ML/VMBench](https://github.com/AMAP-ML/VMBench), закреплённый commit `9772be365784bc3f641d64fca018b1debb749116`.

В этой версии официальный VMBench по-прежнему клонируется setup-методом. Подключение patched fork как submodule — следующий отдельный этап. Существующие точечные исправления и численные ограничения описаны в notebook; исходный runner при извлечении в `utils.py` сохранён без изменения.

Evaluator’ы используют отдельные Python 3.11 environments, не меняют torch notebook и освобождают GPU память при завершении процесса. Веса скачиваются только при запуске нужной метрики. Cache: `/content/VMBench/.cache`, результаты и логи: `/content/vmbench_runs`.

**Полный inference с checkpoint’ами на Colab T4 не проверен.** CPU-путь метрик пока не реализован: при отсутствии GPU выдаётся N/A. MSS FP16 требует около 15.28 GiB только на веса и не помещается в T4; default — teacher CSV, экспериментальная альтернатива — `mss_mode="8bit_experimental"`. CAS checkpoint занимает около 14.2 GB на диске. Неудачный запуск не считается плохим score; изучайте diagnostics.

Результаты сохраняются в `vmbench_results.csv`, `vmbench_diagnostics.csv` и `vmbench_precomputed_export.csv`. Teacher CSV требует совпадения video hash, metric, VMBench commit и adapter profile. Последующий рефакторинг должен расширить идентификацию metadata и parameters; текущий формат сохранён.

## Проверка helpers

```bash
python -m unittest discover -s tests -v
```

Базовые проверки используют стандартную библиотеку и не требуют весов/GPU. Дополнительные проверки таблиц требуют pandas, видео — FFmpeg/ffprobe. Проверка рефакторинга также включает JSON schema notebook, исполнение всех учебных code cells на CPU с заранее предоставленной средой и сравнение исходного/вынесенного runner. Установка в свежем Colab и полный GPU inference остаются отдельным preflight перед занятием.
