
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRIMARY_KEYS = ["TERM_CODE", "SUBJECT_ID_SORT"]
TARGET_COL = "HIGH_ENROLLMENT"
SEASON_ORDER = {"JA": 0, "SP": 1, "SU": 2, "FA": 3}

# Resolve data root: ./input/table_splits/train (agent layout) or BEAVER tree.
_DATA_ROOT_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "input"),
    "/Users/USER/Documents/UNI/SS26/BEAVER",
]
BASE_DATA_DIR = next(
    (
        root
        for root in _DATA_ROOT_CANDIDATES
        if os.path.isdir(os.path.join(root, "table_splits", "train"))
    ),
    os.path.join(SCRIPT_DIR, "input"),
)
TRAIN_DATA_DIR = os.path.join(BASE_DATA_DIR, "table_splits", "train")
GOLD_ENROLLMENT_TRAIN_PATH = os.path.join(TRAIN_DATA_DIR, "gold_enrollment_train.csv")
GOLD_ENROLLMENT_TEST_PATH = os.path.join(BASE_DATA_DIR, "eval", "gold_enrollment_test.csv")

# Feature tables for held-out terms (SUBJECT_SUMMARY.csv, etc.)
_test_default = os.path.join(BASE_DATA_DIR, "test")
TEST_DATA_DIR = _test_default if os.path.isdir(_test_default) else None
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "submission.csv")


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["TERM_CODE"] = out["TERM_CODE"].astype(str).str.strip()
    out["SUBJECT_ID_SORT"] = out["SUBJECT_ID_SORT"].astype(str).str.strip()
    return out


def term_sort_key(term_code: str) -> tuple[int, int]:
    term_code = str(term_code).strip()
    year = int(term_code[:4])
    season = term_code[4:]
    return year, SEASON_ORDER.get(season, 99)


def load_offering_tables(
    data_dir: str, extra_tables: bool = False, keys: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Load one row per offering from SUBJECT_SUMMARY (optionally filtered to keys)."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    summary_path = os.path.join(data_dir, "SUBJECT_SUMMARY.csv")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"SUBJECT_SUMMARY.csv not found in {data_dir}")

    key_index = None
    if keys is not None:
        keys = normalize_keys(keys)[PRIMARY_KEYS].drop_duplicates()
        key_index = set(zip(keys["TERM_CODE"], keys["SUBJECT_ID_SORT"]))

    if key_index is None:
        merged = normalize_keys(pd.read_csv(summary_path, low_memory=False))
    else:
        parts = []
        for chunk in pd.read_csv(summary_path, chunksize=100_000, low_memory=False):
            chunk = normalize_keys(chunk)
            mask = [
                (t, s) in key_index
                for t, s in zip(chunk["TERM_CODE"], chunk["SUBJECT_ID_SORT"])
            ]
            filtered = chunk.loc[mask]
            if not filtered.empty:
                parts.append(filtered)
        if not parts:
            raise ValueError(f"No SUBJECT_SUMMARY rows matched keys in {data_dir}")
        merged = pd.concat(parts, ignore_index=True)

    merged = merged.drop_duplicates(subset=PRIMARY_KEYS, keep="first")

    if not extra_tables:
        return merged

    for name in sorted(os.listdir(data_dir)):
        if not name.endswith(".csv"):
            continue
        if name in {"SUBJECT_SUMMARY.csv", "gold_enrollment_train.csv"}:
            continue
        path = os.path.join(data_dir, name)
        try:
            df = pd.read_csv(path, low_memory=False)
        except pd.errors.EmptyDataError:
            continue
        if not all(key in df.columns for key in PRIMARY_KEYS):
            continue
        df = normalize_keys(df).drop_duplicates(subset=PRIMARY_KEYS, keep="first")
        merged = pd.merge(merged, df, on=PRIMARY_KEYS, how="left", suffixes=("", f"_{name[:-4]}"))

    return merged


def split_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    exclude = set(PRIMARY_KEYS + [TARGET_COL])
    categorical, numerical = [], []
    for col in df.columns:
        if col in exclude:
            continue
        if df[col].dtype == "object" or df[col].nunique(dropna=False) < 50:
            categorical.append(col)
        else:
            numerical.append(col)
    return numerical, categorical


def fit_preprocess(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict, dict, LabelEncoder]:
    """Fit encoders/imputers on training data and return processed frame."""
    df = df.copy()
    target_encoder = LabelEncoder()
    df["HIGH_ENROLLMENT_ENCODED"] = target_encoder.fit_transform(
        df[TARGET_COL].astype(str)
    )

    numerical, categorical = split_feature_columns(df)
    label_encoders: dict[str, LabelEncoder] = {}
    numerical_means: dict[str, float] = {}

    for col in categorical:
        df[col] = df[col].astype(str).fillna("nan_category")
        le = LabelEncoder()
        le.fit(df[col].unique())
        df[col] = le.transform(df[col])
        label_encoders[col] = le

    for col in numerical:
        mean_val = float(df[col].mean()) if df[col].notna().any() else 0.0
        df[col] = df[col].fillna(mean_val)
        numerical_means[col] = mean_val

    feature_columns = numerical + categorical
    return df, feature_columns, label_encoders, numerical_means, target_encoder


def apply_preprocess(
    df: pd.DataFrame,
    feature_columns: list[str],
    label_encoders: dict[str, LabelEncoder],
    numerical_means: dict[str, float],
) -> pd.DataFrame:
    """Apply training-fitted transforms to validation or test rows."""
    out = normalize_keys(df)
    _, categorical = split_feature_columns(
        out.assign(**{TARGET_COL: "N"}) if TARGET_COL not in out.columns else out
    )

    for col in label_encoders:
        le = label_encoders[col]
        oov = len(le.classes_)
        if col in out.columns:
            values = out[col].astype(str).fillna("nan_category")

            def encode_value(x: str, encoder: LabelEncoder = le, fallback: int = oov) -> int:
                if x in encoder.classes_:
                    return int(encoder.transform([x])[0])
                return fallback

            out[col] = values.map(encode_value)
        else:
            out[col] = oov

    for col in numerical_means:
        if col in out.columns:
            out[col] = out[col].fillna(numerical_means[col])
        else:
            out[col] = numerical_means[col]

    for col in feature_columns:
        if col not in out.columns:
            out[col] = 0
    return out[feature_columns]


def time_based_split(
    df: pd.DataFrame, feature_columns: list[str], val_fraction: float = 0.2
):
    ordered_terms = sorted(df["TERM_CODE"].unique(), key=term_sort_key)
    n_val = max(1, int(len(ordered_terms) * val_fraction))
    val_terms = set(ordered_terms[-n_val:])

    train_mask = ~df["TERM_CODE"].isin(val_terms)
    val_mask = df["TERM_CODE"].isin(val_terms)

    X_train = df.loc[train_mask, feature_columns]
    y_train = df.loc[train_mask, "HIGH_ENROLLMENT_ENCODED"]
    X_val = df.loc[val_mask, feature_columns]
    y_val = df.loc[val_mask, "HIGH_ENROLLMENT_ENCODED"]
    return X_train, X_val, y_train, y_val, ordered_terms, sorted(val_terms)


def main() -> None:
    print("Loading training data...")
    if not os.path.isdir(TRAIN_DATA_DIR):
        raise FileNotFoundError(
            f"Training directory not found: {TRAIN_DATA_DIR}. "
            "Expected table_splits/train under ./input or BEAVER root."
        )

    train_features = load_offering_tables(TRAIN_DATA_DIR)
    gold_labels = normalize_keys(pd.read_csv(GOLD_ENROLLMENT_TRAIN_PATH))
    print(f"Loaded features: {train_features.shape}, gold labels: {gold_labels.shape}")

    train_df = pd.merge(
        train_features,
        gold_labels,
        on=PRIMARY_KEYS,
        how="inner",
    )
    if train_df.empty:
        raise ValueError("No training rows after merging features with gold labels.")

    train_df, feature_columns, label_encoders, numerical_means, target_encoder = (
        fit_preprocess(train_df)
    )
    print(f"Combined training data shape: {train_df.shape}")
    print(f"Features used ({len(feature_columns)}): {feature_columns[:10]}...")

    X_train, X_val, y_train, y_val, all_terms, val_terms = time_based_split(
        train_df, feature_columns
    )
    print(f"Total unique terms: {len(all_terms)}")
    print(f"Validation terms ({len(val_terms)}): {val_terms}")
    print(f"Training size: {len(X_train)}, validation size: {len(X_val)}")

    if X_val.empty or y_val.empty:
        print("Warning: empty validation slice; using random split.")
        X_train, X_val, y_train, y_val = train_test_split(
            train_df[feature_columns],
            train_df["HIGH_ENROLLMENT_ENCODED"],
            test_size=0.2,
            random_state=42,
            stratify=train_df["HIGH_ENROLLMENT_ENCODED"],
        )

    print("Training RandomForestClassifier model...")
    model = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred_val = model.predict(X_val)
    final_validation_score = f1_score(y_val, y_pred_val, average="macro")
    print(f"Final Validation Performance: {final_validation_score}")

    # Retrain on all labeled train rows before test inference.
    print("Retraining on full training data...")
    model.fit(train_df[feature_columns], train_df["HIGH_ENROLLMENT_ENCODED"])

    if TEST_DATA_DIR is None and not os.path.isfile(GOLD_ENROLLMENT_TEST_PATH):
        print("No test feature directory or gold_enrollment_test.csv found; skipping test predictions.")
        print("Script finished.")
        return

    test_keys = None
    test_gold = None
    if os.path.isfile(GOLD_ENROLLMENT_TEST_PATH):
        test_gold = normalize_keys(pd.read_csv(GOLD_ENROLLMENT_TEST_PATH))
        test_keys = test_gold[PRIMARY_KEYS].drop_duplicates().reset_index(drop=True)
        print(f"Loaded {len(test_keys)} test keys from {GOLD_ENROLLMENT_TEST_PATH}")

    if TEST_DATA_DIR is not None:
        print(f"Loading test features from {TEST_DATA_DIR}...")
        test_features = load_offering_tables(
            TEST_DATA_DIR, keys=test_keys if test_keys is not None else None
        )
        if test_keys is not None:
            test_features = pd.merge(test_keys, test_features, on=PRIMARY_KEYS, how="left")
        else:
            test_keys = test_features[PRIMARY_KEYS].copy()
    elif test_keys is not None:
        # Keys only: predict with zero/imputed features if test tables are unavailable.
        print("Warning: TEST_DATA_DIR missing; using test keys without feature tables.")
        test_features = test_keys.copy()
    else:
        print("Script finished.")
        return

    X_test = apply_preprocess(test_features, feature_columns, label_encoders, numerical_means)

    print(f"Predicting {len(X_test)} test offerings...")
    y_pred_encoded = model.predict(X_test)
    y_pred = target_encoder.inverse_transform(y_pred_encoded)

    submission = test_keys.copy()
    submission[TARGET_COL] = y_pred
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(submission)} predictions to {OUTPUT_PATH}")

    if test_gold is not None and TARGET_COL in test_gold.columns:
        eval_df = submission.merge(test_gold, on=PRIMARY_KEYS, suffixes=("_pred", "_true"))
        if not eval_df.empty:
            y_true = target_encoder.transform(eval_df[f"{TARGET_COL}_true"].astype(str))
            y_pred_eval = target_encoder.transform(eval_df[f"{TARGET_COL}_pred"].astype(str))
            test_f1 = f1_score(y_true, y_pred_eval, average="macro")
            print(f"Test macro F1 (local, using gold_enrollment_test.csv): {test_f1:.4f}")

    print("Script finished.")


if __name__ == "__main__":
    main()
