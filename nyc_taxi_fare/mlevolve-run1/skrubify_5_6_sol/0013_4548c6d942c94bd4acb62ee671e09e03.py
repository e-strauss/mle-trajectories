import numpy as np
import pandas as pd
import stratum as skrub
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import BaseCrossValidator, train_test_split


class PerChunkSampler(TransformerMixin, BaseEstimator):
    """Reproduce per-read-chunk sampling and the accumulated-row cutoff."""

    def __init__(
        self,
        chunk_column="_source_chunk",
        max_rows_per_chunk=400_000,
        max_total_rows=15_000_000,
        random_state=42,
    ):
        self.chunk_column = chunk_column
        self.max_rows_per_chunk = max_rows_per_chunk
        self.max_total_rows = max_total_rows
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        sampled_chunks = []
        total_rows = 0

        for chunk_id in pd.unique(X[self.chunk_column]):
            chunk = X.loc[X[self.chunk_column] == chunk_id]
            chunk = chunk.drop(columns=[self.chunk_column])

            if len(chunk) > self.max_rows_per_chunk:
                chunk = chunk.sample(
                    n=self.max_rows_per_chunk,
                    random_state=self.random_state,
                )

            sampled_chunks.append(chunk)
            total_rows += len(chunk)
            if total_rows >= self.max_total_rows:
                break

        if not sampled_chunks:
            return X.drop(columns=[self.chunk_column]).reset_index(drop=True)

        return pd.concat(sampled_chunks, ignore_index=True)


class OriginalShuffledHoldout(BaseCrossValidator):
    """Reproduce pandas shuffle followed by the positional 80/20 split."""

    def __init__(self, train_fraction=0.8, random_state=42):
        self.train_fraction = train_fraction
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        indices = np.arange(len(X))
        shuffled = (
            pd.Series(indices)
            .sample(frac=1.0, random_state=self.random_state)
            .to_numpy()
        )
        split_idx = int(len(shuffled) * self.train_fraction)
        yield shuffled[:split_idx], shuffled[split_idx:]


class OptionalUnitCountDistance(TransformerMixin, BaseEstimator):
    """Preserve the original schema-dependent unit_count interaction."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X.copy()
        if "unit_count" in result.columns:
            result["unit_count_x_dist"] = (
                result["unit_count"] * result["euclidean_dist"]
            )
        return result


class Float32Finite(TransformerMixin, BaseEstimator):
    """Match values.astype(np.float32) and np.nan_to_num for features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=np.float32)
        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return pd.DataFrame(values, columns=X.columns, index=X.index)


class GetXY(TransformerMixin, BaseEstimator):
    """Carve an early-stopping set from the current outer training fold."""

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        X_fit, X_val, y_fit, y_val = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        # Match the original conversion of both fitting targets to finite
        # float32 values. This happens only after the raw target is marked.
        y_fit_values = np.nan_to_num(
            np.asarray(y_fit, dtype=np.float32).reshape(-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        y_val_values = np.nan_to_num(
            np.asarray(y_val, dtype=np.float32).reshape(-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        y_fit = pd.Series(
            y_fit_values,
            index=X_fit.index,
            name=getattr(y, "name", None),
        )
        y_val = pd.Series(
            y_val_values,
            index=X_val.index,
            name=getattr(y, "name", None),
        )

        return {
            "X": X_fit,
            "X_val": X_val,
            "y": y_fit,
            "y_val": y_val,
        }

    def transform(self, X):
        return {"X": X, "X_val": None, "y": None, "y_val": None}


def restore_raw_target_domain(values, mode):
    """Gate post-prediction handling because fit mode holds an estimator."""
    if mode == "fit":
        return values

    # The target preprocessing only converts already-filtered positive finite
    # costs to float32, so there is no nonlinear transform to invert.
    return values


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — recorded read. Test-set processing, parquet files,
    #    submission generation, directory creation, and progress printing are
    #    omitted because they do not contribute to the validation score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # Reconstruct the boundaries produced by read_csv(chunksize=2_000_000).
    data = data.assign(_source_chunk=data.index // 2_000_000)

    # 2. Prepare data. Row filtering and row sampling must happen before the
    #    marks because they determine which observations are scored.
    filtered = data[data["cost"] > 0]
    filtered = filtered[filtered["cost"] < 50_000]

    origin_x = filtered["origin_x"].skb.apply_func(
        pd.to_numeric, errors="coerce"
    )
    filtered = filtered.assign(origin_x=origin_x)
    filtered = filtered[
        (filtered["origin_x"] >= -180)
        & (filtered["origin_x"] <= 180)
    ]

    origin_y = filtered["origin_y"].skb.apply_func(
        pd.to_numeric, errors="coerce"
    )
    filtered = filtered.assign(origin_y=origin_y)
    filtered = filtered[
        (filtered["origin_y"] >= -180)
        & (filtered["origin_y"] <= 180)
    ]

    dest_x = filtered["dest_x"].skb.apply_func(
        pd.to_numeric, errors="coerce"
    )
    filtered = filtered.assign(dest_x=dest_x)
    filtered = filtered[
        (filtered["dest_x"] >= -180)
        & (filtered["dest_x"] <= 180)
    ]

    dest_y = filtered["dest_y"].skb.apply_func(
        pd.to_numeric, errors="coerce"
    )
    filtered = filtered.assign(dest_y=dest_y)
    filtered = filtered[
        (filtered["dest_y"] >= -180)
        & (filtered["dest_y"] <= 180)
    ]

    filtered = filtered.skb.apply(
        PerChunkSampler(
            chunk_column="_source_chunk",
            max_rows_per_chunk=400_000,
            max_total_rows=15_000_000,
            random_state=42,
        )
    )

    # Mark the RAW target. The custom splitter reproduces the original pandas
    # shuffle and positional 80/20 holdout.
    y = filtered["cost"].skb.mark_as_y()
    X = filtered.drop(columns=["cost", "record_id"]).skb.mark_as_X(
        cv=OriginalShuffledHoldout(
            train_fraction=0.8,
            random_state=42,
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering.
    start_time = X["start_time"].skb.apply_func(
        pd.to_datetime,
        errors="coerce",
        utc=True,
    )
    hour = start_time.dt.hour
    dayofweek = start_time.dt.dayofweek

    X_feat = X.assign(
        start_time=start_time,
        hour=hour,
        dayofweek=dayofweek,
        month=start_time.dt.month,
        year=start_time.dt.year,
        day=start_time.dt.day,
        hour_sin=(2 * np.pi * hour / 24.0).skb.apply_func(np.sin),
        hour_cos=(2 * np.pi * hour / 24.0).skb.apply_func(np.cos),
        dow_sin=(2 * np.pi * dayofweek / 7.0).skb.apply_func(np.sin),
        dow_cos=(2 * np.pi * dayofweek / 7.0).skb.apply_func(np.cos),
    )

    origin_x = (
        X_feat["origin_x"]
        .skb.apply_func(pd.to_numeric, errors="coerce")
        .clip(-180, 180)
        .fillna(0)
    )
    origin_y = (
        X_feat["origin_y"]
        .skb.apply_func(pd.to_numeric, errors="coerce")
        .clip(-180, 180)
        .fillna(0)
    )
    dest_x = (
        X_feat["dest_x"]
        .skb.apply_func(pd.to_numeric, errors="coerce")
        .clip(-180, 180)
        .fillna(0)
    )
    dest_y = (
        X_feat["dest_y"]
        .skb.apply_func(pd.to_numeric, errors="coerce")
        .clip(-180, 180)
        .fillna(0)
    )

    dx = dest_x - origin_x
    dy = dest_y - origin_y
    euclidean_dist = (dx**2 + dy**2).skb.apply_func(np.sqrt)

    X_feat = X_feat.assign(
        origin_x=origin_x,
        origin_y=origin_y,
        dest_x=dest_x,
        dest_y=dest_y,
        origin_x_bin=origin_x.skb.apply_func(np.round, 2),
        origin_y_bin=origin_y.skb.apply_func(np.round, 2),
        dest_x_bin=dest_x.skb.apply_func(np.round, 2),
        dest_y_bin=dest_y.skb.apply_func(np.round, 2),
        dx=dx,
        dy=dy,
        euclidean_dist=euclidean_dist,
        manhattan_dist=(
            dx.skb.apply_func(np.abs)
            + dy.skb.apply_func(np.abs)
        ),
        bearing=dy.skb.apply_func(np.arctan2, dx),
    )
    X_feat = X_feat.skb.apply(OptionalUnitCountDistance())

    # start_time and record_id were excluded by the original feature_cols list;
    # record_id was already removed when X was constructed.
    X_feat = X_feat.drop(columns=["start_time"])
    X_numeric = X_feat.skb.apply(Float32Finite())

    # The original early-stopped on the same holdout rows it reported, which
    # leaks validation information. Keep n_estimators=1000, the 20-round
    # patience, and a 20% eval fraction, but carve that eval set only from each
    # outer fold's training observations.
    X_y = X_numeric.skb.apply(
        GetXY(test_size=0.2, random_state=42),
        y=y,
        how="no_wrap",
    )
    X_fit = X_y["X"]
    y_fit = X_y.get("y", y)
    X_val = X_y["X_val"]
    y_val = X_y["y_val"]

    model = lgb.LGBMRegressor(
        n_estimators=1000,
        num_leaves=127,
        learning_rate=0.03,
        min_child_samples=50,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
    )

    # log_evaluation only printed progress and is omitted. Early stopping keeps
    # the original stopping_rounds=20.
    pred_transformed = X_fit.skb.apply(
        model,
        y=y_fit,
        fit_kwargs={
            "eval_set": [(X_val, y_val)],
            "callbacks": [
                lgb.early_stopping(
                    stopping_rounds=20,
                    verbose=False,
                )
            ],
        },
    )

    # The fitted target passed through the original float32/finite conversion.
    # Return predictions to the raw mark_as_y scoring path, gated so fit mode
    # leaves the fitted estimator unchanged.
    pred = pred_transformed.skb.apply_func(
        restore_raw_target_domain,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= here — the splitter on mark_as_X drives.
    if __name__ == "__main__":
        skrub.set_config(scheduler=True, debug_graph=True)
        search = pred.skb.make_grid_search(
            n_jobs=1,
            fitted=True,
            refit=False,
            scoring="neg_root_mean_squared_error",
        )
        print(search.results_)
        for variant_score in search.results_["mean_test_score"]:
            print(f"Variant score: {variant_score}")
        print(
            "Final Validation Performance: "
            f"{search.results_['mean_test_score'].iloc[0]}"
        )
