# Implementation notes

Detailed adapter changes and validation limits retained from the original workshop notebook.

### Исправления интеграции upstream

Это адаптер проверенной ревизии, не новое API VMBench. Оригинальные файлы не перезаписываются.
Адаптер в `utils.py` применяет патчи к копии текста в дочернем процессе; проверяем, что ожидаемая строка присутствует ровно один раз.

1. **PAS:** `apply_boxes_torch` получает `(C,H)` из normalized tensor вместо `(H,W)` исходного изображения.
   Исправляем размер на `image_array.shape[:2]`. Это исправление масок **может изменить score** относительно buggy CLI.
   Сумма смещений, нормализация диагональю и правило компенсации камеры сохранены. PAS здесь сырой,
   не ограничен единицей; обещать шкалу 0–100 по описанию paper нельзя.
2. **GroundingDINO:** запускаем уже включённую в проект PyTorch реализацию deformable attention на CUDA,
   когда `_C` не собран. Алгоритм не заменяем; fused kernel и fallback не обязаны быть побитово равны.
3. **OIS:** без person bbox MMPose может перейти к whole-image pose; без instances `pose_utils` обращается
   к `instances[0]`. Вместо такого результата возвращаем N/A, если хотя бы один кадр не содержит валидную
   human detection. Это политика применимости, не заполнение отсутствующих кадров выдуманными позами.
4. **TCS:** выключаем безусловный BF16 autocast (FP32 на всех GPU для воспроизводимости с T4,
   возможно немного другие числа),
   включаем официальный `offload_state_to_cpu=True`. Число SAM2 instances сохраняем как диагностику.
   Без detected objects upstream может вернуть 1.0; сохраняем score, но явно предупреждаем.
   Если SAM2 CUDA extension не собрался, его официальный fallback пропускает small-hole postprocessing;
   этот запуск явно помечается и не является точной репликацией CUDA pipeline.
5. **CAS:** текущий `commonsense_adherence_score.py` вообще не читает `--finetune`.
   Создаём **официальный** `vit_giant_patch14_224` напрямую (обход `pretrained_cfg` несовместимого timm factory),
   загружаем checkpoint `module` с `strict=True`, используем официальный dataset, `final_test`, `final_merge`.
   Сохраняем 16 frames, sampling rate 4, 10 temporal segments ×3 crops; batch=1, workers=1.
   Merge делаем последовательно вместо Pool(64). DeepSpeed/training не нужны.
6. **MSS:** исходный CLI отправляет все sliding windows одним batch. Вызываем тот же scorer по одному окну
   (размер 3, включая официальный padding), затем те же threshold/artifact функции для полной последовательности.
   При 8-bit меняется точность хранения весов, что записывается в profile/source.

CSV считывает настоящие JSON keys напрямую: upstream `calculate_score.py` ошибочно ищет
`perceptible_amplitude_socre`. Для leaderboard сравнение адаптированных запусков требует отдельной валидации.

### Память, ошибки, таймер и заранее рассчитанные scores

Каждый evaluator запускается отдельным процессом. После выхода процесса все его CUDA allocations
освобождаются; `empty_cache()` в notebook не умеет освобождать память другого живого процесса.
При таймауте дочерний процесс завершается. Полный traceback находится в `metric.log`, короткое объяснение — в таблице.
Runtime inference отделён от setup/download; cache hit имеет Runtime=0, исходное время сохраняется отдельно.

Для fallback преподаватель может разместить CSV по `config.precomputed_url` или сохранить его по `workshop.precomputed_path`.
Обязательные колонки: `video_sha256,metric,score,runtime_sec,vmbench_commit,adapter_profile`.
Hash вычисляется по **конкретному файлу**; другой encode не подойдёт. Только teacher scores с той же
ревизией и profile принимаются автоматически. Никаких демонстрационных чисел не выдумываем.
Score заранее рассчитанного запуска явно маркируется `precomputed`, runtime исходного запуска не выдаётся за время Colab.

## 13. Контроль памяти и таблица для одного видео

В notebook остаются только числа и пути. Модели создавались в отдельных процессах и после выхода исчезли.
`peak_allocated_gib` в diagnostics — фактический PyTorch peak дочернего процесса, если inference завершился.
Он не включает всю память CUDA context и других процессов. Заранее точный peak для T4 неизвестен.

Ориентиры, **не GPU-замеры**: OIS обычно самый лёгкий; PAS — несколько GB весов плюс CoTracker activations;
TCS — примерно 2.5 GB FP32 весов плюс две SAM2 активации/маски/CoTracker; CAS — 3.77 GiB FP32 весов
плюс attention activations; MSS FP16 — ≥15.28 GiB только веса, ожидайте GPU 24 GB или больше.
Для коротких входов целевой бюджет PAS/OIS/TCS/CAS — до 15 GiB, гарантия требует Colab preflight.
MSS 8-bit может уложиться в T4, но этот inference здесь не проверен.
