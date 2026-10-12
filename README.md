# VMBench workshop

1. [Open the notebook in Google Colab](https://colab.research.google.com/github/KK-one/vmbench-workshop/blob/main/vmbench_workshop_colab.ipynb).
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Configure your videos and metrics in the configuration cell.
4. Select **Runtime → Run all**. The first cell downloads the repo and its submodule; upload your videos when prompted.
5. Download `/content/vmbench_results.csv`.

Both repositories must be public. After a runtime reset, run all cells and upload your videos again. For MSS on T4, provide the instructor's precomputed CSV or explicitly select the experimental `mss_mode="8bit_experimental"`.
