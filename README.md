# VMBench workshop

1. [Join the Kaggle competition](https://www.kaggle.com/t/ac5704b939234a8d96a93d8aabc66bde) and accept its rules.
2. [Open in Colab](https://colab.research.google.com/github/KK-one/vmbench-workshop/blob/main/vmbench_workshop_colab.ipynb). Select **T4 GPU** for live metrics.
3. Run the notebook. Enter the competition slug when prompted and provide your personal [Kaggle API token](https://www.kaggle.com/settings/api), using the Colab secret `KAGGLE_API_TOKEN` or masked input. Videos download and unpack automatically.
4. Browse videos. Optionally compute **PAS/OIS/TCS/CAS** for selected IDs (two videos by default). Set `RUN_METRICS=False` for video review only. MSS is excluded on T4.
5. Edit `baseline_all_real.csv`: **0 = real, 1 = generated**. Upload the CSV with columns `video_id,label` to Kaggle. Scoring is accuracy; all correct labels earn 100%.

After a runtime reset, run all cells again. Videos, credentials and ground-truth files are excluded from Git.
