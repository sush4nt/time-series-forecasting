"""Part B — Classical ML baseline forecasting pipeline.

A small, linear pipeline for benchmarking tree-based demand-forecasting models:

    1. load        -> src.data.load_data
    2. features    -> src.features.build_features
    3. split       -> src.splits.make_splits
    4. train       -> src.models.train_model
    5. evaluate    -> src.evaluate.evaluate_run

Run it end-to-end via ``train.py`` (thin CLI over ``src.pipeline.run``).
"""
