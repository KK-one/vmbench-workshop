# VMBench workshop

1. [Open the notebook in Google Colab](https://colab.research.google.com/github/KK-one/vmbench-workshop/blob/main/vmbench_workshop_colab.ipynb).
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Choose your metrics in **0. Initialization**.
4. Select **Runtime → Run all**. The first code cell clones the repo with its submodule; upload videos when prompted.
5. Download `/content/vmbench_results.csv`.

Local `vids/` is excluded from Git. After a runtime reset, run all cells again. For MSS on T4, provide the instructor's precomputed CSV or select the experimental `mss_mode="8bit_experimental"`.
