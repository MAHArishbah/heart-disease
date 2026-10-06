"""Heart-disease risk model, refactored out of ClassificationPrediction.ipynb.

Module map (see docs/notebook-refactor-map.pdf for the cell-by-cell version):

    config.py    cell 0        paths, params.yaml loader, seed, cv
    data.py      cells 1,8,13,23,24    load_raw / clean / select_features / split
    features.py  cell 28       build_preprocessor
    train.py     cells 37,51,56        grid search, cost threshold, artifacts
    evaluate.py  cells 34,72,76        score helper, test-set report
    serve.py     (new)         FastAPI prediction endpoint

The rule that keeps this honest: notebooks import from here, never the reverse.
"""

__all__ = ["config", "data", "features", "train", "evaluate","tracking","logging_conf"]
