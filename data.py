"""Load bikeshare, drop leakage/correlated columns, log1p target, split."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config as C


def load():
    df = pd.read_csv(C.DATA_PATH)
    y_raw = df[C.TARGET].to_numpy(dtype=float)
    y = np.log1p(y_raw) if C.LOG1P_TARGET else y_raw

    X = df.drop(columns=[C.TARGET] + C.DROP)
    # CatBoost wants categorical columns as int (not float) — they already are.
    for c in C.CAT_FEATURES:
        X[c] = X[c].astype(int)
    return X, y, y_raw


def split(X, y, y_raw):
    idx = np.arange(len(y))
    Xtr, Xte, ytr, yte, itr, ite = train_test_split(
        X, y, idx, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE
    )
    return {
        "X_train": Xtr.reset_index(drop=True), "y_train": ytr,
        "X_test": Xte.reset_index(drop=True),  "y_test": yte,
        "idx_train": itr, "idx_test": ite,
        "y_raw_train": y_raw[itr], "y_raw_test": y_raw[ite],
    }


def correlation_report(X):
    """Print pairwise correlation of the kept features (sanity check)."""
    corr = X.corr(numeric_only=True)
    hi = (corr.abs() > 0.8) & (corr.abs() < 1.0)
    pairs = [(a, b, corr.loc[a, b]) for a in corr.index for b in corr.columns
             if a < b and hi.loc[a, b]]
    print("Kept features:", list(X.columns))
    if pairs:
        print("Remaining |corr|>0.8 pairs:")
        for a, b, r in pairs:
            print(f"  {a:12s} {b:12s} {r:+.3f}")
    else:
        print("No remaining |corr|>0.8 feature pairs.")
    return corr


if __name__ == "__main__":
    X, y, y_raw = load()
    correlation_report(X)
    d = split(X, y, y_raw)
    print(f"train {d['X_train'].shape}  test {d['X_test'].shape}")
