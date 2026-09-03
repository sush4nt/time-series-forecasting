"""FMCG demand forecasting — classical ML + deep learning pipelines.

One unified runner benchmarks tree models and a GRU encoder-decoder on the same
splits and metrics:

    1. load        -> src.data.load_data
    2. prepare     -> backend (features+splits  OR  sequence windows)
    3. fit         -> backend (tree training     OR  GRU training loop)
    4. predict     -> backend (per-split meta + predictions)
    5. evaluate    -> src.evaluate.evaluate_split

Run it end-to-end via ``train.py`` (thin CLI over ``src.runner.run``).
"""
