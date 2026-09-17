import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import BaseCrossValidator, train_test_split


class ChunkSampleHoldout(BaseCrossValidator):
    """Reproduce the original chunk caps, row limit, shuffle, and 80/20 split."""

    def __init__(
        self,
        max_rows_per_chunk=400_000,
        total_row_limit=10_000_000,
        train_fraction=0.8,
        random_state=42,
    ):
        self.max_rows_per_chunk = max_rows_per_chunk
        self.total_row_limit = total_row_limit
        self.train_fraction = train_fraction
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("ChunkSampleHoldout requires chunk IDs via groups.")

        chunk_ids = np.asarray(groups).reshape(-1)
        selected_parts = []
        total_rows = 0

        # The original restarted random_state=42 for every oversized CSV chunk.
        for chunk_id in pd.unique(chunk_ids):
            chunk_positions = np.flatnonzero(chunk_ids == chunk_id)

            if len(chunk_positions) > self.max_rows_per_chunk:
                chunk_positions = (
                    pd.Series(chunk_positions)
                    .sample(
                        n=self.max_rows_per_chunk,
                        random_state=self.random_state,
                    )
                    .to_numpy()
                )

            selected_parts.append(chunk_positions)
            total_rows += len(chunk_positions)

            # As in the original, the chunk that crosses the limit is retained
            # in full, so the selected count may exceed total_row_limit.
            if total_rows >= self.total_row_limit:
                break

        selected = np.concatenate(selected_parts)
        shuffled = (
            pd.Series(selected)
            .sample(frac=1.0, random_state=self.random_state)
            .to_numpy()
        )

        split_idx = int(len(shuffled) * self.train_fraction)
        yield shuffled[:split_idx], shuffled[split_idx:]


class GetXY(TransformerMixin, BaseEstimator):
    """Carve an early-stopping set out of this outer fold's training rows."""

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        parts = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        return dict(zip(("X", "X_val", "y", "y_val"), parts))

    def transform(self, X):
        return {"X": X, "X_val": None, "y": None, "y_val": None}


def restore_target_domain(values, mode):
    """Return predictions in the raw cost domain used by mark_as_y."""
    if mode == "fit":
        # In fit mode a prediction node evaluates to the fitted estimator.
        return values
    # Casting the training target to float32 did not change its units or apply a
    # nonlinear transform; conversion back to float64 is its prediction-side
    # inverse for scoring against the raw target node.
    return np.asarray(values, dtype=np.float64)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record one read of train.csv. Test-set processing,
    #    parquet files, submission generation, directories, and progress output
    #    are omitted because they do not contribute to the validation score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # Preserve the original 2,000,000-row CSV chunk boundaries. The custom CV
    # splitter below performs the original per-chunk sampling, cumulative
    # 10,000,000-row stopping rule, shuffle, and final holdout without chunked IO.
    data = data.assign(_chunk=data.index // 2_000_000)

    # 2. Prepare Data — faithfully record the content-dependent row filters
    #    before marking X and y because these filters change which rows are scored.
    data = data[data["cost"] > 0]
    data = data[data["cost"] < 50_000]

    origin_x = data["origin_x"].skb.apply_func(pd.to_numeric, errors="coerce")
    data = data.assign(origin_x=origin_x)
    data = data[(data["origin_x"] >= -180) & (data["origin_x"] <= 180)]

    origin_y = data["origin_y"].skb.apply_func(pd.to_numeric, errors="coerce")
    data = data.assign(origin_y=origin_y)
    data = data[(data["origin_y"] >= -180) & (data["origin_y"] <= 180)]

    dest_x = data["dest_x"].skb.apply_func(pd.to_numeric, errors="coerce")
    data = data.assign(dest_x=dest_x)
    data = data[(data["dest_x"] >= -180) & (data["dest_x"] <= 180)]

    dest_y = data["dest_y"].skb.apply_func(pd.to_numeric, errors="coerce")
    data = data.assign(dest_y=dest_y)
    data = data[(data["dest_y"] >= -180) & (data["dest_y"] <= 180)]
    data = data.reset_index(drop=True)

    # Mark the RAW target. The custom splitter reproduces the original sampled,
    # shuffled 80/20 holdout and excludes rows beyond the original chunk limit.
    y = data["cost"].skb.mark_as_y()
    X = data.drop(columns=["cost"]).skb.mark_as_X(
        cv=ChunkSampleHoldout(
            max_rows_per_chunk=400_000,
            total_row_limit=10_000_000,
            train_fraction=0.8,
            random_state=42,
        ),
        split_kwargs={"groups": data["_chunk"]},
    )

    # 3. Recorded feature engineering, preserving the original named columns
    #    and their order.
    start_time = X["start_time"].skb.apply_func(
        pd.to_datetime,
        errors="coerce",
        utc=True,
    )
    X_feat = X.assign(
        start_time=start_time,
        hour=start_time.dt.hour,
        dayofweek=start_time.dt.dayofweek,
        month=start_time.dt.month,
        year=start_time.dt.year,
        day=start_time.dt.day,
    )

    origin_x = X_feat["origin_x"].skb.apply_func(pd.to_numeric, errors="coerce")
    origin_y = X_feat["origin_y"].skb.apply_func(pd.to_numeric, errors="coerce")
    dest_x = X_feat["dest_x"].skb.apply_func(pd.to_numeric, errors="coerce")
    dest_y = X_feat["dest_y"].skb.apply_func(pd.to_numeric, errors="coerce")

    X_feat = X_feat.assign(
        origin_x=origin_x.clip(-180, 180).fillna(0),
        origin_y=origin_y.clip(-180, 180).fillna(0),
        dest_x=dest_x.clip(-180, 180).fillna(0),
        dest_y=dest_y.clip(-180, 180).fillna(0),
    )

    dx = X_feat["dest_x"] - X_feat["origin_x"]
    dy = X_feat["dest_y"] - X_feat["origin_y"]
    X_feat = X_feat.assign(
        dx=dx,
        dy=dy,
        euclidean_dist=(dx**2 + dy**2).skb.apply_func(np.sqrt),
        manhattan_dist=dx.skb.apply_func(np.abs) + dy.skb.apply_func(np.abs),
    )

    # Match feature_cols and the original float32/nan_to_num conversion. The
    # helper _chunk column is used only by the splitter and never by the model.
    X_feat = X_feat.drop(columns=["record_id", "start_time", "_chunk"])
    X_feat = (
        X_feat.astype(np.float32)
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
    )

    # The original explicitly converted y to float32 and applied nan_to_num.
    # Keep that target preprocessing downstream of the raw mark, then restore
    # prediction values to the raw target domain before scoring.
    y_model = (
        y.astype(np.float32)
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
    )

    # The original early-stopped on the same validation rows it scored, which
    # leaks and cannot be reproduced honestly under outer validation. Keep its
    # 20-round patience, 300-estimator limit, and 20% split fraction by carving
    # the early-stopping set from each outer fold's training rows.
    X_y = X_feat.skb.apply(
        GetXY(test_size=0.2, random_state=42),
        y=y_model,
        how="no_wrap",
    )
    X_fit = X_y["X"]
    y_fit = X_y.get("y", y_model)
    X_val = X_y["X_val"]
    y_val = X_y["y_val"]

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
    )
    pred_model_domain = X_fit.skb.apply(
        model,
        y=y_fit,
        fit_kwargs={
            "eval_set": [(X_val, y_val)],
            "callbacks": [lgb.early_stopping(20, verbose=False)],
        },
    )

    # Restore predictions after fitting on the transformed target. The eval-mode
    # guard leaves the fitted estimator untouched in fit mode.
    pred = pred_model_domain.skb.apply_func(
        restore_target_domain,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= here — the splitter attached to mark_as_X drives.
    if __name__ == "__main__":
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
