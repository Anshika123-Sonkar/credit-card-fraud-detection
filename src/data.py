from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(path, max_rows=None, random_state=42):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Download the public CSV and save it as data/creditcard.csv.")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset file is empty: {path}. Download and extract the real creditcard.csv file.") from exc
    if df.empty or "Class" not in df.columns:
        raise ValueError("Dataset must be a non-empty CSV containing a 'Class' target column.")
    df = df.drop_duplicates().dropna(subset=["Class"]).copy()
    y = pd.to_numeric(df.pop("Class"), errors="coerce")
    X = df.apply(pd.to_numeric, errors="coerce").replace([float("inf"), float("-inf")], float("nan"))
    valid = y.notna() & X.notna().all(axis=1)
    X, y = X.loc[valid], y.loc[valid].astype(int)
    if set(y.unique()) != {0, 1}:
        raise ValueError("Class must contain both labels 0 and 1, where 1 means fraud.")
    if X.shape[1] == 0:
        raise ValueError("Dataset has no usable feature columns besides 'Class'.")
    if max_rows and len(X) > max_rows:
        X, _, y, _ = train_test_split(X, y, train_size=max_rows, stratify=y, random_state=random_state)
    if y.value_counts().min() < 5:
        raise ValueError("Need at least 5 examples of each class.")
    return X.reset_index(drop=True), y.reset_index(drop=True)

def split_dataset(X, y, random_state=42):
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=.4, stratify=y, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=.5, stratify=y_tmp, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test
