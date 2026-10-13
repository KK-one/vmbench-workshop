# VMBench workshop

1. [Join the Kaggle competition](https://www.kaggle.com/t/ac5704b939234a8d96a93d8aabc66bde), accept its rules and download the videos ZIP from **Data** (or **Download All**).
2. [Open in Colab](https://colab.research.google.com/github/KK-one/vmbench-workshop/blob/main/vmbench_workshop_colab.ipynb). Select **T4 GPU** for live metrics.
3. Upload the ZIP to `/content` using **Files → Upload**, wait for completion, then select **Runtime → Run all**. The notebook clones the code and unpacks the videos. No API token is needed.
4. Browse videos. Optionally compute **PAS/OIS/TCS/CAS** for selected IDs (two videos by default). Set `RUN_METRICS=False` for video review only. MSS is excluded on T4.
5. Edit `baseline_all_real.csv`: **0 = real, 1 = generated**. Upload the CSV with columns `video_id,label` to Kaggle. Scoring is accuracy; all correct labels earn 100%.

After a runtime reset, upload the ZIP and run all cells again. Videos and ground-truth files are excluded from Git.

After metric setup, call the evaluators directly:

```python
pas = compute_pas(video_path, subject="dog", box_threshold=0.25, text_threshold=0.20, grid_size=30)
ois = compute_ois(video_path, bbox_thr=0.3, nms_thr=0.3)
tcs = compute_tcs(video_path, subject="dog", box_threshold=0.25, text_threshold=0.20, grid_size=50)
cas = compute_cas(video_path, test_num_segment=10, test_num_crop=3, num_workers=1)
```

`subject` is English detection text; OIS evaluates people. Changing detector/tracker settings or CAS test views may change scores. Settings are recorded in diagnostics and included in the cache key.
