import pandas as pd
import skrub
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score

TRAIN_PATH = "gs://mle-trajectories-data/nyc_taxi_fare/input/train.csv.gz"
TEST_PATH = "gs://mle-trajectories-data/nyc_taxi_fare/input/test.csv"
# Subsampling 500k rows deterministically from the 55.4M training set provides
# comprehensive NYC spatial and temporal coverage while ensuring 5-fold CV fits
# run quickly within the experiment timeout and compute budget.
N_ROWS = 500_000


class AdversarialClassifier(BaseEstimator):
    def __init__(self, max_iter: int = 200):
        self.max_iter = max_iter

    def fit(self, X, y):
        self.model_ = LogisticRegression(max_iter=self.max_iter)
        scores = cross_val_score(self.model_, X, y, cv=3, scoring="roc_auc")
        self.mean_auc_ = float(scores.mean())
        self.std_auc_ = float(scores.std())
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def transform(self, X):
        return pd.DataFrame({
            "adversarial_roc_auc": [self.mean_auc_] * len(X),
            "adversarial_roc_auc_std": [self.std_auc_] * len(X),
        })


def build():
    train = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv, nrows=N_ROWS)
    test = skrub.as_data_op(TEST_PATH).skb.apply_func(pd.read_csv)

    # Filter out corrupted coordinates, non-positive fares, and invalid passenger counts
    valid_mask = (
        (train["fare_amount"] > 0)
        & (train["fare_amount"] <= 500)
        & (train["pickup_longitude"] >= -75.0)
        & (train["pickup_longitude"] <= -72.0)
        & (train["pickup_latitude"] >= 40.0)
        & (train["pickup_latitude"] <= 42.0)
        & (train["dropoff_longitude"] >= -75.0)
        & (train["dropoff_longitude"] <= -72.0)
        & (train["dropoff_latitude"] >= 40.0)
        & (train["dropoff_latitude"] <= 42.0)
        & (train["passenger_count"] >= 1)
        & (train["passenger_count"] <= 6)
    ).fillna(False)

    clean_train = train[valid_mask].reset_index(drop=True)

    # Adversarial check between cleaned train population and test population
    feature_cols = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
    train_adv = clean_train[feature_cols].head(10000).assign(is_test=0)
    test_adv = test[feature_cols].assign(is_test=1)
    combined = train_adv.skb.concat([test_adv], axis=0).reset_index(drop=True)
    adv_X = combined[feature_cols]
    adv_y = combined["is_test"]
    adv_check = adv_X.skb.apply(AdversarialClassifier(), y=adv_y)

    # Separate X and target y, then mark them
    X_raw = clean_train.drop(columns=["fare_amount"])
    y_raw = clean_train["fare_amount"]

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    X = X_raw.skb.mark_as_X(cv=cv, split_kwargs={})
    y = y_raw.skb.mark_as_y()

    audit = {
        "raw_train_rows": train.skb.apply_func(len),
        "clean_train_rows": clean_train.skb.apply_func(len),
        "test_rows": test.skb.apply_func(len),
        "adversarial_check": adv_check,
    }

    return {
        "X": X,
        "y": y,
        "scoring": "neg_root_mean_squared_error",
        "audit": audit,
    }