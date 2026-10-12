# VMBench workshop

1. [Open the notebook in Google Colab](https://colab.research.google.com/github/KK-one/vmbench-workshop/blob/main/vmbench_workshop_colab.ipynb).
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Prepare an existing checkout at `/content/workshop` with its `vendor/vmbench` submodule.
4. Configure **0. Initialization**, then select **Runtime → Run all**. Use local `vids/` or upload videos when prompted.
5. Download `/content/vmbench_results.csv`.

The notebook does not clone repositories. Local `vids/` is excluded from Git. After a runtime reset, prepare the checkout again. For MSS on T4, provide the instructor's precomputed CSV or select the experimental `mss_mode="8bit_experimental"`.
