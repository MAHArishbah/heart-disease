"""Load, clean, select, split

Every transformation here is a *deterministic rule* -- it uses no quantity
estimated from the data, so running it before the train/test split leaks
nothing. Anything that learns a parameter (medians, category lists, scaling)
is in features.py, inside the Pipeline, so that it runs at predict time
with the training values instead of being recomputed on incoming rows.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import PARAMS, PROJECT_ROOT, SEED


COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "num"]


def load_raw(raw_dir: Path | str | None = None) -> pd.DataFrame:
    """Concatenate the four site files, tagging each row with its site."""
    cfg = PARAMS["data"]
    raw_dir = Path(raw_dir) if raw_dir else PROJECT_ROOT / cfg["raw_dir"]

    frames = []
    for site, filename in cfg["files"].items():
        frame = pd.read_csv(raw_dir / filename, names=COLS, na_values="?")
        frame["site"] = site
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def apply_zero_rules(df: pd.DataFrame) -> pd.DataFrame:
    """ change the physiscally impossible zero values like chol and heartrate to nan. kept out of clean as this needs to run while serving as well"""
    df = df.copy()
    for col in PARAMS["data"]["zero_is_missing"]:
        if col in df.columns:
            df.loc[df[col] == 0, col] = np.nan
    return df


def clean(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Impossible zeros -> NaN, binarise target, drop duplicates. drop duplicates flag to keep df unchanged if needed for eda"""
    df = apply_zero_rules(df)

    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns="num")

    return df.drop_duplicates(keep="first") if drop_duplicates else df


def select_features(df: pd.DataFrame, roles: dict | None = None,verbose:bool=False) -> dict:
    """Resolve the declared roles into concrete column lists.Only training set to be used to avoid leakage """
    cfg = PARAMS["features"]
    roles = {k: list(v) for k, v in (roles or cfg["roles"]).items()}

    if cfg["drop_high_missing"]:
        thresh = cfg["missing_threshold"]
        high_miss = [c for c in sum(roles.values(),[]) if df[c].isna().mean() > thresh]
        if verbose:
            print(f'dropping high miss columns enabled with threshold {thresh}, col to be dropped {high_miss if high_miss else None}')
        roles = {k: [c for c in v if c not in high_miss] for k, v in roles.items()}
    else:
        high_miss = []

    numeric = roles["cont"] + roles["ordinal"]
    binary = roles["binary"]
    categorical = roles["nominal"] + (["site"] if cfg["include_site"] else [])
    declared=set(sum(roles.values(),[])) | set (high_miss)
    unknown=[c for c in cfg['missing_indicators'] if c not in declared]
    indicators=[c for c in cfg["missing_indicators"] if c in numeric + binary]
    if unknown:
        raise ValueError(f'missing indicator columns with no role {unknown}')
    if verbose and cfg["include_site"] :
        print(f'site included in the features')

    return {
        "cont": roles["cont"],
        "ordinal": roles["ordinal"],
        "numeric": numeric,
        "binary": binary,
        "categorical": categorical,
        "features": numeric + binary + categorical,
        "dropped_high_missing": high_miss,
        "indicators":indicators
    }


def split(df: pd.DataFrame):
    """Stratified split on the target.this also returns `g_test` if using LOGO"""

    y = df["target"]
    groups = df["site"]
    X = df.drop(columns=["target"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=PARAMS["data"]["test_size"],
        stratify=y,
        random_state=SEED,
    )
    return (X_train, X_test, y_train, y_test,
            groups.loc[X_train.index], groups.loc[X_test.index])


def prepare(df: pd.DataFrame | None = None,verbose:bool=False) -> dict:
    """load_raw -> clean -> split -> select_features, in the leakage-safe order.the canonical entry point: train.py, evaluate.py and the notebooks all call
    this, so they cannot drift apart on which rows or columns they are using.
    """
    if df is None:
        if verbose:
            print('applyig sentinel value correction , target binarization and feature selection on training df')
        df = clean(load_raw())

    X_train, X_test, y_train, y_test, g_train, g_test = split(df)
    spec = select_features(X_train)          

    return {
        "df": df,
        "X_train": X_train[spec["features"]].copy(),
        "X_test": X_test[spec["features"]].copy(),
        "y_train": y_train,
        "y_test": y_test,
        "g_train": g_train,
        "g_test": g_test,
        "spec": spec,
    }
