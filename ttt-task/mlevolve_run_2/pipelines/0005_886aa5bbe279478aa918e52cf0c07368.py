import copy
import gc
import json
import os
import re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix, save_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------------------------------------------------
# Global Configuration and Seed Setup
# -------------------------------------------------------------------------
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------------
# Step 1: Data Processing and Feature Engineering
# -------------------------------------------------------------------------
# 1. Load Trackers Metadata and Encode Tracker-Level Semantic Attributes
trackers_df = pd.read_csv(os.path.join(INPUT_DIR, "trackers.tsv"), sep="\t")
trackers_df["tracker_id"] = trackers_df["tracker_id"].astype(np.int32)
trackers_df["tracking_domain_id"] = trackers_df["tracking_domain_id"].astype(np.int64)
trackers_df = trackers_df.sort_values("tracker_id").reset_index(drop=True)
trackers_df.to_parquet(
    os.path.join(WORKING_DIR, "tracker_metadata.parquet"), index=False
)
tracker_domain_ids_set = set(trackers_df["tracking_domain_id"].unique())
num_trackers = len(trackers_df)


# Encode categorical metadata attributes (category, country, brand) into integer index tensors
def encode_meta_column(col_series):
    vals = col_series.fillna("").astype(str).str.strip()
    vals = vals.replace({"#": "", "nan": "", "None": ""})
    unique_vals = sorted([v for v in vals.unique() if v != ""])
    val_to_idx = {v: i + 1 for i, v in enumerate(unique_vals)}  # Index 0 reserved for unknown
    encoded = np.array([val_to_idx.get(v, 0) for v in vals], dtype=np.int64)
    return encoded, len(unique_vals) + 1


cat_indices, num_categories = encode_meta_column(trackers_df["category"])
country_indices, num_countries = encode_meta_column(trackers_df["country"])
brand_indices, num_brands = encode_meta_column(trackers_df["brand"])

tracker_meta_indices = {
    "category": torch.from_numpy(cat_indices).long(),
    "country": torch.from_numpy(country_indices).long(),
    "brand": torch.from_numpy(brand_indices).long(),
    "num_categories": num_categories,
    "num_countries": num_countries,
    "num_brands": num_brands,
}

# 2. Load Target (Test) Domains
target_df = pd.read_csv(os.path.join(INPUT_DIR, "target.tsv"), sep="\t")
test_domain_ids = target_df["domain_id"].astype(np.int64).values
test_domains_set = set(test_domain_ids)
np.save(os.path.join(WORKING_DIR, "test_domain_ids.npy"), test_domain_ids)

# 3. Load Tracking Graph & Create Train/Validation Split
tg_table = pq.read_table(
    os.path.join(INPUT_DIR, "tracking_graph_train.parquet"),
    columns=["domain_id", "tracker_id"],
)
tg_domain_ids = tg_table["domain_id"].to_numpy()
tg_tracker_ids = tg_table["tracker_id"].to_numpy().astype(np.int32)
del tg_table
gc.collect()

# Exclude any test domains from training candidates
unique_tg_domains = np.unique(tg_domain_ids)
valid_candidate_domains = unique_tg_domains[
    ~np.isin(unique_tg_domains, test_domain_ids)
]

# Shuffle candidates deterministically
shuffled_candidates = np.random.RandomState(SEED).permutation(valid_candidate_domains)
n_candidates = len(shuffled_candidates)
val_size = min(30000, int(n_candidates * 0.15))
train_size = min(200000, n_candidates - val_size)

val_domain_ids = shuffled_candidates[:val_size]
train_domain_ids = shuffled_candidates[val_size : val_size + train_size]

np.save(os.path.join(WORKING_DIR, "val_domain_ids.npy"), val_domain_ids)
np.save(os.path.join(WORKING_DIR, "train_domain_ids.npy"), train_domain_ids)


# Fast CSR ground truth matrix construction
def build_ground_truth_csr(domain_ids_arr, tg_doms, tg_tracks, num_tr):
    dom_to_idx = {d: i for i, d in enumerate(domain_ids_arr)}
    mask = np.isin(tg_doms, domain_ids_arr)
    sub_doms = tg_doms[mask]
    sub_tracks = tg_tracks[mask]

    rows = np.array([dom_to_idx[d] for d in sub_doms], dtype=np.int32)
    cols = sub_tracks
    vals = np.ones(len(rows), dtype=np.uint8)

    csr = csr_matrix(
        (vals, (rows, cols)), shape=(len(domain_ids_arr), num_tr), dtype=np.uint8
    )
    csr.data = np.minimum(csr.data, 1)
    return csr


val_labels_csr = build_ground_truth_csr(
    val_domain_ids, tg_domain_ids, tg_tracker_ids, num_trackers
)
train_labels_csr = build_ground_truth_csr(
    train_domain_ids, tg_domain_ids, tg_tracker_ids, num_trackers
)

save_npz(os.path.join(WORKING_DIR, "val_labels.npz"), val_labels_csr)
save_npz(os.path.join(WORKING_DIR, "train_labels.npz"), train_labels_csr)

# Global tracker frequencies strictly on training domains
train_tracker_counts = np.asarray(train_labels_csr.sum(axis=0)).ravel()
global_tracker_prior = train_tracker_counts / len(train_domain_ids)
top_trackers_global = np.argsort(-global_tracker_prior)[:10]

# Compute Tracker Co-occurrence Matrix (355 x 355) strictly on training set
co_occ = (train_labels_csr.T @ train_labels_csr).astype(np.float32).toarray()
co_occ_norm = co_occ / (train_tracker_counts[:, None] + 1e-6)
np.savez_compressed(
    os.path.join(WORKING_DIR, "tracker_priors.npz"),
    global_prior=global_tracker_prior,
    co_occurrence=co_occ_norm,
)

# Compute symmetric normalized tracker co-occurrence adjacency matrix with self-loops strictly from training labels
co_occ_self = co_occ + np.eye(num_trackers, dtype=np.float32)
deg = co_occ_self.sum(axis=1)
deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-8))
norm_co_occ_np = deg_inv_sqrt[:, None] * co_occ_self * deg_inv_sqrt[None, :]
norm_co_occurrence = torch.from_numpy(norm_co_occ_np.astype(np.float32))

del tg_domain_ids, tg_tracker_ids
gc.collect()

# 4. Load Hostname Strings for Required Domains
all_needed_ids = set(train_domain_ids) | set(val_domain_ids) | set(test_domain_ids)
domains_table = pq.read_table(
    os.path.join(INPUT_DIR, "domains.parquet"),
    columns=["domain_id", "domain"],
)
dom_ids = domains_table["domain_id"].to_numpy()
dom_names = domains_table["domain"].to_numpy()
del domains_table
gc.collect()

mask_needed = np.isin(dom_ids, list(all_needed_ids))
needed_dom_ids = dom_ids[mask_needed]
needed_dom_names = dom_names[mask_needed]
del dom_ids, dom_names
gc.collect()

domain_lookup = dict(zip(needed_dom_ids, needed_dom_names))
del needed_dom_ids, needed_dom_names
gc.collect()

# 5. Load Freedom of the Press Dataset
press_freedom_map = {}
press_path = os.path.join(INPUT_DIR, "freedom-of-the-press.csv")
if os.path.exists(press_path):
    try:
        df_press = pd.read_csv(press_path, sep="\t")
        if len(df_press.columns) == 1:
            df_press = pd.read_csv(press_path, sep=",")
    except Exception:
        df_press = pd.read_csv(press_path)
    df_press.columns = [c.strip() for c in df_press.columns]
    for _, row in df_press.iterrows():
        tld_val = str(row["tld"]).strip().lower()
        try:
            score_val = float(row["freedom_of_the_press"])
            press_freedom_map[tld_val] = score_val
        except (ValueError, TypeError):
            continue

# 6. Load and Aggregate URL Classification Categories
url_cats_df = pd.read_csv(
    os.path.join(INPUT_DIR, "url-classification.csv"),
    usecols=["url", "category"],
)
known_categories = sorted(url_cats_df["category"].dropna().unique().tolist())
cat_to_idx = {cat: i for i, cat in enumerate(known_categories)}
url_domain_pattern = re.compile(r"^(?:https?://)?(?:www\.)?([^/:\?]+)", re.IGNORECASE)


def clean_url_to_domain(u):
    if not isinstance(u, str):
        return ""
    m = url_domain_pattern.match(u.strip())
    return m.group(1).lower() if m else ""


url_cats_df["clean_domain"] = url_cats_df["url"].apply(clean_url_to_domain)
url_cats_df = url_cats_df[url_cats_df["clean_domain"] != ""]

# Aggregate category counts per domain string
grouped = url_cats_df.groupby(["clean_domain", "category"]).size().unstack(fill_value=0)
domain_category_map = grouped.to_dict(orient="index")
del url_cats_df, grouped
gc.collect()

# 7. Load Link Graph Centrality & Tracker Outlink Features
lg_table = pq.read_table(
    os.path.join(INPUT_DIR, "link-graph.parquet"),
    columns=["source_domain_id", "target_domain_id"],
)
src_arr = lg_table["source_domain_id"].to_numpy()
dst_arr = lg_table["target_domain_id"].to_numpy()
del lg_table
gc.collect()

# Direct links to trackers
tracker_dst_mask = np.isin(
    dst_arr, np.array(list(tracker_domain_ids_set), dtype=np.int64)
)
tracker_src_ids = src_arr[tracker_dst_mask]
tracker_dst_ids = dst_arr[tracker_dst_mask]

# Top 10 most linked-to tracker domains
top_linked_tracker_domains = (
    pd.Series(tracker_dst_ids).value_counts().head(10).index.tolist()
)

# Out-degree and in-degree Series
out_degrees = pd.Series(src_arr).value_counts()
in_degrees = pd.Series(dst_arr).value_counts()
tracker_out_degrees = pd.Series(tracker_src_ids).value_counts()

# Top tracker specific link indicators
tracker_link_pairs = pd.DataFrame({"src": tracker_src_ids, "dst": tracker_dst_ids})
tracker_link_pairs = tracker_link_pairs[
    tracker_link_pairs["dst"].isin(top_linked_tracker_domains)
].drop_duplicates()
tracker_pair_set = set(zip(tracker_link_pairs["src"], tracker_link_pairs["dst"]))

# Precompute target domain index mappings
all_domain_ids = np.concatenate([train_domain_ids, val_domain_ids, test_domain_ids])
N_eval = len(all_domain_ids)
N_train = len(train_domain_ids)
eval_dom_to_idx = {d: i for i, d in enumerate(all_domain_ids)}
train_dom_to_idx = {d: i for i, d in enumerate(train_domain_ids)}

# Structural Block 1: Full 355-Tracker Direct Link Indicators
tracker_domain_to_id = dict(
    zip(trackers_df["tracking_domain_id"].values, trackers_df["tracker_id"].values)
)
sub_tr_mask = np.isin(tracker_src_ids, all_domain_ids)
sub_tr_src = tracker_src_ids[sub_tr_mask]
sub_tr_dst = tracker_dst_ids[sub_tr_mask]
sub_tr_rows = np.array([eval_dom_to_idx[d] for d in sub_tr_src], dtype=np.int32)
sub_tr_cols = np.array([tracker_domain_to_id[d] for d in sub_tr_dst], dtype=np.int32)
direct_links_mat = np.zeros((N_eval, num_trackers), dtype=np.float32)
direct_links_mat[sub_tr_rows, sub_tr_cols] = 1.0
del sub_tr_mask, sub_tr_src, sub_tr_dst, sub_tr_rows, sub_tr_cols

# Structural Block 2: 1-Hop Out-Neighbor Training Tracker Consensus
src_in_eval = np.isin(src_arr, all_domain_ids)
dst_in_train = np.isin(dst_arr, train_domain_ids)
out_mask = src_in_eval & dst_in_train & (src_arr != dst_arr)
del src_in_eval, dst_in_train
gc.collect()

out_src = src_arr[out_mask]
out_dst = dst_arr[out_mask]
del out_mask
gc.collect()

row_out = np.array([eval_dom_to_idx[d] for d in out_src], dtype=np.int32)
col_out = np.array([train_dom_to_idx[d] for d in out_dst], dtype=np.int32)
del out_src, out_dst
gc.collect()

A_out = csr_matrix(
    (np.ones(len(row_out), dtype=np.float32), (row_out, col_out)),
    shape=(N_eval, N_train),
    dtype=np.float32,
)
del row_out, col_out
gc.collect()

A_out.data = np.minimum(A_out.data, 1.0)
deg_out = np.asarray(A_out.sum(axis=1)).ravel()
inv_deg_out = np.zeros(N_eval, dtype=np.float32)
pos_out = deg_out > 0
inv_deg_out[pos_out] = 1.0 / deg_out[pos_out]

sum_y_out = A_out @ train_labels_csr.astype(np.float32)
consensus_out_mat = (
    sum_y_out.toarray() if hasattr(sum_y_out, "toarray") else sum_y_out
) * inv_deg_out[:, None]
del A_out, sum_y_out, deg_out, inv_deg_out
gc.collect()

# Structural Block 3: 1-Hop In-Neighbor Training Tracker Consensus
src_in_train = np.isin(src_arr, train_domain_ids)
dst_in_eval = np.isin(dst_arr, all_domain_ids)
in_mask = src_in_train & dst_in_eval & (src_arr != dst_arr)
del src_in_train, dst_in_eval
gc.collect()

in_src = src_arr[in_mask]
in_dst = dst_arr[in_mask]
del in_mask
gc.collect()

row_in = np.array([eval_dom_to_idx[d] for d in in_dst], dtype=np.int32)
col_in = np.array([train_dom_to_idx[d] for d in in_src], dtype=np.int32)
del in_src, in_dst
gc.collect()

A_in = csr_matrix(
    (np.ones(len(row_in), dtype=np.float32), (row_in, col_in)),
    shape=(N_eval, N_train),
    dtype=np.float32,
)
del row_in, col_in
gc.collect()

A_in.data = np.minimum(A_in.data, 1.0)
deg_in = np.asarray(A_in.sum(axis=1)).ravel()
inv_deg_in = np.zeros(N_eval, dtype=np.float32)
pos_in = deg_in > 0
inv_deg_in[pos_in] = 1.0 / deg_in[pos_in]

sum_y_in = A_in @ train_labels_csr.astype(np.float32)
consensus_in_mat = (
    sum_y_in.toarray() if hasattr(sum_y_in, "toarray") else sum_y_in
) * inv_deg_in[:, None]
del A_in, sum_y_in, deg_in, inv_deg_in
gc.collect()

del src_arr, dst_arr, tracker_src_ids, tracker_dst_ids, tracker_link_pairs
gc.collect()


# 8. Helper Functions for Feature Extraction
def extract_tld(d_str):
    if not isinstance(d_str, str) or "." not in d_str:
        return "unknown"
    parts = d_str.lower().strip().split(".")
    if (
        len(parts) >= 3
        and parts[-2]
        in {
            "co",
            "com",
            "org",
            "net",
            "edu",
            "gov",
            "ac",
            "mil",
            "gv",
            "ne",
        }
        and len(parts[-1]) == 2
    ):
        return f"{parts[-2]}.{parts[-1]}"
    return parts[-1]


# Extract TLDs for train set and full evaluation set
all_hostnames = [domain_lookup.get(did, "") for did in all_domain_ids]
all_tlds = [extract_tld(h) for h in all_hostnames]
train_hostnames = all_hostnames[: len(train_domain_ids)]
train_tlds = all_tlds[: len(train_domain_ids)]
tld_freq_map = pd.Series(train_tlds).value_counts(normalize=True).to_dict()
top_25_tlds = pd.Series(train_tlds).value_counts().head(25).index.tolist()

# Compute TLD-conditional tracker rates strictly on training set
train_tld_df = pd.DataFrame(
    {"tld": train_tlds, "row_idx": np.arange(len(train_domain_ids))}
)
tld_tracker_probs = {}
tld_expected_count = {}
for tld_name, group in train_tld_df.groupby("tld"):
    if len(group) >= 15:
        group_rows = group["row_idx"].values
        sub_labels = train_labels_csr[group_rows]
        mean_probs = np.asarray(sub_labels.mean(axis=0)).ravel()
        tld_tracker_probs[tld_name] = mean_probs
        tld_expected_count[tld_name] = float(sub_labels.sum(axis=1).mean())

# Structural Block 4: Full 355-dim Smoothed TLD-Conditional Tracker Likelihoods
train_tld_series = pd.Series(train_tlds)
tld_to_train_indices = train_tld_series.groupby(train_tld_series).indices
alpha_tld = 10.0
tld_smoothed_prior_map = {}
for tld_name, train_idx_list in tld_to_train_indices.items():
    n_tld = len(train_idx_list)
    sub_labels = train_labels_csr[train_idx_list]
    count_tld = np.asarray(sub_labels.sum(axis=0)).ravel().astype(np.float32)
    smoothed_prob = (count_tld + alpha_tld * global_tracker_prior) / (
        n_tld + alpha_tld
    )
    tld_smoothed_prior_map[tld_name] = smoothed_prob.astype(np.float32)

tld_prior_mat = np.zeros((N_eval, num_trackers), dtype=np.float32)
for i, t in enumerate(all_tlds):
    tld_prior_mat[i] = tld_smoothed_prior_map.get(t, global_tracker_prior)

# Assemble four 355-dim tracker-aligned structural blocks (1420 dimensions total)
structural_features_all = np.hstack(
    [
        direct_links_mat,
        consensus_out_mat,
        consensus_in_mat,
        tld_prior_mat,
    ]
).astype(np.float32)

struct_feature_names = (
    [f"direct_link_tr_{k}" for k in range(num_trackers)]
    + [f"out_neighbor_consensus_tr_{k}" for k in range(num_trackers)]
    + [f"in_neighbor_consensus_tr_{k}" for k in range(num_trackers)]
    + [f"tld_prior_tr_{k}" for k in range(num_trackers)]
)

del direct_links_mat, consensus_out_mat, consensus_in_mat, tld_prior_mat
gc.collect()

train_press_scores = [
    press_freedom_map.get(t, np.nan) for t in train_tlds if t in press_freedom_map
]
median_press_score = (
    float(np.nanmedian(train_press_scores)) if len(train_press_scores) > 0 else 25.0
)

# 9. Subword Character N-Gram SVD Representation
tfidf = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 4),
    min_df=5,
    max_features=5000,
    dtype=np.float32,
)
svd = TruncatedSVD(n_components=32, random_state=SEED)
train_tfidf = tfidf.fit_transform(train_hostnames)
svd.fit(train_tfidf)
del train_tfidf
gc.collect()


# 10. Vectorized Feature Extraction Routine
def extract_feature_matrix(domain_id_list, hostnames_list):
    n = len(domain_id_list)
    features_dict = {}

    # Lexical features
    lens = np.array([len(h) for h in hostnames_list], dtype=np.float32)
    dots = np.array([h.count(".") for h in hostnames_list], dtype=np.float32)
    hyphens = np.array([h.count("-") for h in hostnames_list], dtype=np.float32)
    digits = np.array(
        [sum(c.isdigit() for c in h) for h in hostnames_list], dtype=np.float32
    )
    vowels = np.array(
        [sum(c in "aeiou" for c in h.lower()) for h in hostnames_list],
        dtype=np.float32,
    )

    features_dict["domain_len"] = lens
    features_dict["num_dots"] = dots
    features_dict["num_hyphens"] = hyphens
    features_dict["num_digits"] = digits
    features_dict["digit_ratio"] = digits / (lens + 1.0)
    features_dict["vowel_ratio"] = vowels / (lens + 1.0)

    # Keywords / Intent Indicators
    features_dict["has_www"] = np.array(
        [1.0 if h.lower().startswith("www.") else 0.0 for h in hostnames_list],
        dtype=np.float32,
    )
    features_dict["has_cdn"] = np.array(
        [
            1.0 if any(k in h.lower() for k in ["cdn", "static", "assets"]) else 0.0
            for h in hostnames_list
        ],
        dtype=np.float32,
    )
    features_dict["has_blog"] = np.array(
        [
            1.0 if any(k in h.lower() for k in ["blog", "news", "press"]) else 0.0
            for h in hostnames_list
        ],
        dtype=np.float32,
    )
    features_dict["has_shop"] = np.array(
        [
            (
                1.0
                if any(k in h.lower() for k in ["shop", "store", "market", "cart"])
                else 0.0
            )
            for h in hostnames_list
        ],
        dtype=np.float32,
    )
    features_dict["has_app"] = np.array(
        [
            1.0 if any(k in h.lower() for k in ["api", "app", "dev", "cloud"]) else 0.0
            for h in hostnames_list
        ],
        dtype=np.float32,
    )
    features_dict["has_gov_or_edu"] = np.array(
        [
            1.0 if any(k in h.lower() for k in [".gov", ".edu", ".ac."]) else 0.0
            for h in hostnames_list
        ],
        dtype=np.float32,
    )

    # TLD features
    tlds = [extract_tld(h) for h in hostnames_list]
    features_dict["is_cctld"] = np.array(
        [1.0 if len(t) == 2 else 0.0 for t in tlds], dtype=np.float32
    )
    features_dict["tld_freq"] = np.array(
        [tld_freq_map.get(t, 0.0) for t in tlds], dtype=np.float32
    )

    for top_tld in top_25_tlds:
        features_dict[f"tld_is_{top_tld}"] = np.array(
            [1.0 if t == top_tld else 0.0 for t in tlds], dtype=np.float32
        )

    # Freedom of the Press features
    press_scores = []
    press_missing = []
    for t in tlds:
        if t in press_freedom_map:
            press_scores.append(press_freedom_map[t])
            press_missing.append(0.0)
        else:
            press_scores.append(median_press_score)
            press_missing.append(1.0)
    features_dict["press_freedom"] = np.array(press_scores, dtype=np.float32)
    features_dict["press_missing"] = np.array(press_missing, dtype=np.float32)

    # URL Classification features
    has_cat = []
    cat_matrix = np.zeros((n, len(known_categories)), dtype=np.float32)
    for i, h in enumerate(hostnames_list):
        h_clean = h.lower().strip()
        if h_clean.startswith("www."):
            h_clean = h_clean[4:]
        if h_clean in domain_category_map:
            has_cat.append(1.0)
            c_dict = domain_category_map[h_clean]
            for cat_name, cnt in c_dict.items():
                if cat_name in cat_to_idx:
                    cat_matrix[i, cat_to_idx[cat_name]] = float(cnt)
        else:
            has_cat.append(0.0)

    features_dict["has_url_category"] = np.array(has_cat, dtype=np.float32)
    for idx_cat, cat_name in enumerate(known_categories):
        features_dict[f"url_cat_{cat_name}"] = cat_matrix[:, idx_cat]

    # Link Graph features
    out_deg = out_degrees.reindex(domain_id_list, fill_value=0).values.astype(
        np.float32
    )
    in_deg = in_degrees.reindex(domain_id_list, fill_value=0).values.astype(np.float32)
    tr_out = tracker_out_degrees.reindex(domain_id_list, fill_value=0).values.astype(
        np.float32
    )

    features_dict["out_degree"] = out_deg
    features_dict["in_degree"] = in_deg
    features_dict["log_out_degree"] = np.log1p(out_deg)
    features_dict["log_in_degree"] = np.log1p(in_deg)
    features_dict["degree_ratio"] = (
        features_dict["log_in_degree"] - features_dict["log_out_degree"]
    )
    features_dict["total_degree"] = np.log1p(out_deg + in_deg)
    features_dict["is_isolated"] = np.array(
        (out_deg == 0) & (in_deg == 0), dtype=np.float32
    )
    features_dict["tracker_outlinks"] = tr_out
    features_dict["has_tracker_outlinks"] = np.array(tr_out > 0, dtype=np.float32)

    # Top tracker specific links
    for tr_dom_id in top_linked_tracker_domains:
        features_dict[f"links_to_tracker_{tr_dom_id}"] = np.array(
            [
                1.0 if (did, tr_dom_id) in tracker_pair_set else 0.0
                for did in domain_id_list
            ],
            dtype=np.float32,
        )

    # TLD Tracker Likelihood Prior features
    mean_expected = float(np.mean(list(tld_expected_count.values()) or [3.0]))
    features_dict["tld_expected_trackers"] = np.array(
        [tld_expected_count.get(t, mean_expected) for t in tlds], dtype=np.float32
    )

    top5_tld_probs = np.zeros((n, 5), dtype=np.float32)
    for i, t in enumerate(tlds):
        p_vec = tld_tracker_probs.get(t, global_tracker_prior)
        top5_tld_probs[i] = p_vec[top_trackers_global[:5]]

    for k in range(5):
        features_dict[f"tld_prior_tracker_{k}"] = top5_tld_probs[:, k]

    # Character N-gram SVD features
    tfidf_mat = tfidf.transform(hostnames_list)
    svd_mat = svd.transform(tfidf_mat).astype(np.float32)
    for k in range(svd_mat.shape[1]):
        features_dict[f"svd_char_{k}"] = svd_mat[:, k]

    ctx_df = pd.DataFrame(features_dict)
    row_idx = np.array(
        [eval_dom_to_idx[did] for did in domain_id_list], dtype=np.int32
    )
    struct_df = pd.DataFrame(
        structural_features_all[row_idx], columns=struct_feature_names
    )
    return pd.concat([ctx_df, struct_df], axis=1)


# Extract Feature Matrices
train_features_df = extract_feature_matrix(train_domain_ids, train_hostnames)

val_hostnames = [domain_lookup.get(did, "") for did in val_domain_ids]
val_features_df = extract_feature_matrix(val_domain_ids, val_hostnames)

test_hostnames = [domain_lookup.get(did, "") for did in test_domain_ids]
test_features_df = extract_feature_matrix(test_domain_ids, test_hostnames)

# Fill residual NaNs with training medians
train_medians = train_features_df.median()
train_features_df.fillna(train_medians, inplace=True)
val_features_df.fillna(train_medians, inplace=True)
test_features_df.fillna(train_medians, inplace=True)

num_features = train_features_df.shape[1]

# Prior log odds for output layer initialization
eps_p = 1e-4
p_clipped = np.clip(global_tracker_prior, eps_p, 1.0 - eps_p)
prior_log_odds = np.log(p_clipped / (1.0 - p_clipped)).astype(np.float32)


# -------------------------------------------------------------------------
# Step 2: Model Architecture Design
# -------------------------------------------------------------------------
class SwishGLUResidualBlock(nn.Module):
    """Gated Linear Unit residual block with Swish activation for tabular representations."""

    def __init__(self, dim: int, expansion_factor: int = 2, dropout: float = 0.15):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        hidden_dim = dim * expansion_factor
        self.linear_gate = nn.Linear(dim, hidden_dim)
        self.linear_val = nn.Linear(dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        gated = F.silu(self.linear_val(x_norm)) * self.linear_gate(x_norm)
        out = self.dropout(self.out_proj(gated))
        return residual + out


class EmpiricalGraphDiffusion(nn.Module):
    """Empirical graph diffusion over normalized tracker co-occurrence adjacency."""

    def __init__(self, norm_co_occ: torch.Tensor, init_alpha: float = 0.1):
        super().__init__()
        self.register_buffer("A_norm", norm_co_occ)
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits + self.alpha * torch.matmul(logits, self.A_norm)


class TrackerGatedRankNet(nn.Module):
    """Graph-Augmented Dual-Stream Network with Tracker Metadata Embeddings and Grouped Channel Projections."""

    def __init__(
        self,
        in_features: int,
        num_tr: int = 355,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.15,
        struct_dropout: float = 0.25,
        num_struct: int = 1420,
        prior_bias: np.ndarray = None,
        tracker_meta_indices: dict = None,
        norm_co_occurrence: torch.Tensor = None,
    ):
        super().__init__()
        self.num_tr = num_tr
        self.num_struct = num_struct
        self.num_ctx = in_features - num_struct
        assert (
            self.num_ctx > 0
        ), f"in_features ({in_features}) must be > num_struct ({num_struct})"

        # 1. Contextual stream
        self.input_bn = nn.BatchNorm1d(self.num_ctx)
        self.input_proj = nn.Linear(self.num_ctx, hidden_dim)
        self.proj_act = nn.SiLU()

        self.blocks = nn.ModuleList(
            [
                SwishGLUResidualBlock(hidden_dim, expansion_factor=2, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.final_dropout = nn.Dropout(dropout)

        # Tracker Metadata Embeddings & Domain-Tracker Dot-Product Affinity
        if tracker_meta_indices is None:
            tracker_meta_indices = {
                "category": torch.zeros(num_tr, dtype=torch.long),
                "country": torch.zeros(num_tr, dtype=torch.long),
                "brand": torch.zeros(num_tr, dtype=torch.long),
                "num_categories": 1,
                "num_countries": 1,
                "num_brands": 1,
            }

        self.register_buffer("meta_category", tracker_meta_indices["category"])
        self.register_buffer("meta_country", tracker_meta_indices["country"])
        self.register_buffer("meta_brand", tracker_meta_indices["brand"])
        self.register_buffer("meta_tracker_ids", torch.arange(num_tr, dtype=torch.long))

        cat_dim = 32
        country_dim = 32
        brand_dim = 64
        id_dim = 64

        self.cat_emb = nn.Embedding(tracker_meta_indices["num_categories"], cat_dim)
        self.country_emb = nn.Embedding(tracker_meta_indices["num_countries"], country_dim)
        self.brand_emb = nn.Embedding(tracker_meta_indices["num_brands"], brand_dim)
        self.id_emb = nn.Embedding(num_tr, id_dim)

        meta_total_dim = cat_dim + country_dim + brand_dim + id_dim
        self.meta_proj = nn.Sequential(
            nn.Linear(meta_total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.tracker_bias = nn.Parameter(torch.zeros(num_tr))
        if prior_bias is not None and len(prior_bias) == num_tr:
            with torch.no_grad():
                self.tracker_bias.copy_(torch.from_numpy(prior_bias))

        # 2. Structural stream: Tracker-aligned grouped channel projection (4 channels -> 1 logit per tracker)
        self.struct_dropout = nn.Dropout(struct_dropout)
        self.struct_weights = nn.Parameter(torch.randn(num_tr, 4) * 0.1)

        # Context-conditioned gated bypass
        self.gate_proj = nn.Linear(hidden_dim, num_tr)

        # 3. Empirical co-occurrence graph diffusion
        if norm_co_occurrence is None:
            norm_co_occurrence = torch.eye(num_tr, dtype=torch.float32)
        self.graph_diffusion = EmpiricalGraphDiffusion(norm_co_occurrence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split into contextual and structural streams
        x_ctx = x[:, : self.num_ctx]
        x_struct = x[:, self.num_ctx :]

        # 1. Contextual stream processing with dot-product affinity
        h = self.input_bn(x_ctx)
        h = self.proj_act(self.input_proj(h))
        for block in self.blocks:
            h = block(h)
        h = self.final_dropout(self.final_norm(h))

        # Build tracker representations from metadata embeddings
        c_e = self.cat_emb(self.meta_category)
        k_e = self.country_emb(self.meta_country)
        b_e = self.brand_emb(self.meta_brand)
        i_e = self.id_emb(self.meta_tracker_ids)
        tracker_meta = torch.cat([c_e, k_e, b_e, i_e], dim=-1)
        tracker_emb = self.meta_proj(tracker_meta)  # (num_tr, hidden_dim)

        logit_ctx = torch.matmul(h, tracker_emb.t()) + self.tracker_bias

        # 2. Structural stream processing with tracker-aligned grouped projection
        # Reshape (batch, 1420) -> (batch, 4, 355) -> (batch, 355, 4)
        x_struct_grouped = x_struct.view(-1, 4, self.num_tr).permute(0, 2, 1)
        x_struct_dropped = self.struct_dropout(x_struct_grouped)
        logit_struct = (x_struct_dropped * self.struct_weights).sum(dim=-1)

        # 3. Context-conditioned gated fusion
        gate = torch.sigmoid(self.gate_proj(h))
        fused_logits = logit_ctx + gate * logit_struct

        # 4. Empirical co-occurrence graph diffusion
        refined_logits = self.graph_diffusion(fused_logits)
        return refined_logits


class TopKAsymmetricRecallLoss(nn.Module):
    """Composite loss combining Asymmetric Focal BCE and Top-10 Hard Negative Rank-Discounted Margin Ranking."""

    def __init__(
        self,
        gamma_neg: float = 2.0,
        margin: float = 1.0,
        top_k_neg: int = 10,
        ranking_weight: float = 0.6,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.margin = margin
        self.top_k_neg = top_k_neg
        self.ranking_weight = ranking_weight
        self.eps = eps

        # Rank discount weights: 1 / log2(rank + 2) for rank in 0 .. top_k_neg - 1
        ranks = torch.arange(top_k_neg, dtype=torch.float32)
        rank_weights = 1.0 / torch.log2(ranks + 2.0)
        normalized_weights = rank_weights / rank_weights.sum()
        self.register_buffer("rank_weights", normalized_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets_f = targets.float()

        # 1. Asymmetric Focal Component
        pos_loss = -targets_f * torch.log(probs + self.eps)
        neg_weights = torch.pow(probs, self.gamma_neg)
        neg_loss = -(1.0 - targets_f) * neg_weights * torch.log(1.0 - probs + self.eps)
        asym_bce = (pos_loss + neg_loss).sum(dim=1).mean()

        # 2. Differentiable Top-K Negative Margin Ranking Component with Rank Discounting
        neg_mask = targets_f == 0.0
        neg_scores = torch.where(
            neg_mask, logits, torch.tensor(-1e6, device=logits.device)
        )
        top_k_val = min(self.top_k_neg, logits.size(1))
        top_negatives, _ = torch.topk(neg_scores, k=top_k_val, dim=1)

        pos_counts = targets_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        has_positives = (targets_f.sum(dim=1) > 0).float()

        diff = self.margin + top_negatives.unsqueeze(1) - logits.unsqueeze(2)
        pairwise_margin = F.softplus(diff)
        pos_mask = targets_f.unsqueeze(2)

        # Apply rank discount weights along top_k_val dimension
        weights = self.rank_weights[:top_k_val].view(1, 1, top_k_val)
        weighted_margin = pairwise_margin * pos_mask * weights

        ranking_loss_per_sample = weighted_margin.sum(dim=(1, 2)) / pos_counts.squeeze(1)
        rank_loss = (ranking_loss_per_sample * has_positives).sum() / (
            has_positives.sum() + self.eps
        )

        return asym_bce + self.ranking_weight * rank_loss


# Instantiate Model, Loss, Optimizer, and Warmup-Cosine Scheduler
model = TrackerGatedRankNet(
    in_features=num_features,
    num_tr=num_trackers,
    hidden_dim=256,
    num_blocks=3,
    dropout=0.15,
    prior_bias=prior_log_odds,
    tracker_meta_indices=tracker_meta_indices,
    norm_co_occurrence=norm_co_occurrence,
).to(device)

criterion = TopKAsymmetricRecallLoss(
    gamma_neg=2.0,
    margin=1.0,
    top_k_neg=10,
    ranking_weight=0.6,
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
    betas=(0.9, 0.999),
)

epochs = 12
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.1,
    end_factor=1.0,
    total_iters=1,
)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs - 1,
    eta_min=1e-5,
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[1],
)

# -------------------------------------------------------------------------
# Step 3: Model Training and Evaluation
# -------------------------------------------------------------------------
X_train_tensor = torch.from_numpy(train_features_df.values.astype(np.float32))
y_train_tensor = torch.from_numpy(train_labels_csr.toarray().astype(np.float32))

X_val_tensor = torch.from_numpy(val_features_df.values.astype(np.float32))
y_val_tensor = torch.from_numpy(val_labels_csr.toarray().astype(np.float32))

X_test_tensor = torch.from_numpy(test_features_df.values.astype(np.float32))

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(
    train_dataset,
    batch_size=512,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
)


def compute_recall_at_10(
    eval_model: torch.nn.Module,
    features_t: torch.Tensor,
    targets_t: torch.Tensor,
    batch_size: int = 1024,
) -> float:
    eval_model.eval()
    all_hits = []
    all_totals = []
    with torch.no_grad():
        n = features_t.size(0)
        for i in range(0, n, batch_size):
            bx = features_t[i : i + batch_size].to(device, non_blocking=True)
            by = targets_t[i : i + batch_size].to(device, non_blocking=True)

            logits = eval_model(bx)
            _, top10_preds = torch.topk(logits, k=10, dim=1)

            hits = torch.gather(by, 1, top10_preds).sum(dim=1).float()
            totals = by.sum(dim=1).float()

            all_hits.append(hits.cpu())
            all_totals.append(totals.cpu())

    hits_vec = torch.cat(all_hits, dim=0)
    totals_vec = torch.cat(all_totals, dim=0)

    domain_recalls = torch.where(
        totals_vec > 0, hits_vec / totals_vec, torch.zeros_like(hits_vec)
    )
    return float(domain_recalls.mean().item())


best_val_recall = -1.0
best_model_weights = None
patience = 4
patience_counter = 0

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    num_batches = 0

    for bx, by in train_loader:
        bx = bx.to(device, non_blocking=True)
        by = by.to(device, non_blocking=True)

        optimizer.zero_grad()
        preds = model(bx)
        loss = criterion(preds, by)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    scheduler.step()
    epoch_loss = running_loss / max(num_batches, 1)

    val_recall = compute_recall_at_10(
        model, X_val_tensor, y_val_tensor, batch_size=1024
    )
    print(
        f"Epoch {epoch:02d}/{epochs:02d} - Loss: {epoch_loss:.4f} - Val Recall@10: {val_recall:.4f}"
    )

    if val_recall > best_val_recall:
        best_val_recall = val_recall
        best_model_weights = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break

# Load Best Model Checkpoint
if best_model_weights is not None:
    model.load_state_dict(best_model_weights)
    torch.save(best_model_weights, os.path.join(WORKING_DIR, "best_tracker_model.pt"))

final_val_score = compute_recall_at_10(
    model, X_val_tensor, y_val_tensor, batch_size=1024
)

# -------------------------------------------------------------------------
# Test Inference and Submission Generation
# -------------------------------------------------------------------------
model.eval()
test_predictions_list = []

with torch.no_grad():
    n_test = X_test_tensor.size(0)
    for i in range(0, n_test, 1024):
        bx = X_test_tensor[i : i + 1024].to(device, non_blocking=True)
        logits = model(bx)
        _, top10_preds = torch.topk(logits, k=10, dim=1)
        test_predictions_list.append(top10_preds.cpu().numpy())

test_pred_tracker_ids = np.concatenate(test_predictions_list, axis=0)

# Build Tracker ID to Tracking Domain ID Lookup
tracker_id_to_domain_id = np.zeros(num_trackers, dtype=np.int64)
tracker_id_to_domain_id[trackers_df["tracker_id"].values] = trackers_df[
    "tracking_domain_id"
].values

# Map predicted tracker indices to full tracking domain IDs
test_pred_tracking_domains = tracker_id_to_domain_id[test_pred_tracker_ids]

# Format submission table
repeated_domain_ids = np.repeat(test_domain_ids, 10)
flat_tracking_domain_ids = test_pred_tracking_domains.ravel()

submission_df = pd.DataFrame(
    {
        "domain_id": repeated_domain_ids,
        "tracking_domain_id": flat_tracking_domain_ids,
    }
)

# Save submission in both CSV and TSV formats
sub_csv_path = os.path.join(SUBMISSION_DIR, "submission.csv")
sub_tsv_path = os.path.join(SUBMISSION_DIR, "submission.tsv")

submission_df.to_csv(sub_csv_path, sep="\t", index=False)
submission_df.to_csv(sub_tsv_path, sep="\t", index=False)

# Clean up memory
del (
    X_train_tensor,
    y_train_tensor,
    X_val_tensor,
    y_val_tensor,
    X_test_tensor,
    train_dataset,
    train_loader,
)
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# -------------------------------------------------------------------------
# Final Score Output
# -------------------------------------------------------------------------
print(f"Final Validation Score: {final_val_score}")
