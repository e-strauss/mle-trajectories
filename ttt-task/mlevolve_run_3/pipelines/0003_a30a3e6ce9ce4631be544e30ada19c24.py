import copy
import json
import math
import os
import sys
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import gc
from torch.utils.data import DataLoader, TensorDataset

# Ensure reproducible operations and directories
script_start_time = time.time()
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    torch.set_num_threads(min(8, os.cpu_count() or 4))

os.makedirs("working", exist_ok=True)
os.makedirs("submission", exist_ok=True)

# ==============================================================================
# 1. LOAD TARGETS AND TRACKER METADATA
# ==============================================================================
# Load target domains (test set)
df_target = pd.read_csv("input/target.tsv", sep="\t")
test_domain_ids = df_target["domain_id"].astype(np.int64).values
test_domain_set = set(test_domain_ids)

# Load tracker metadata
df_trackers = pd.read_csv("input/trackers.tsv", sep="\t")
num_trackers = len(df_trackers)
tracker_id_to_domain = dict(
    zip(df_trackers["tracker_id"], df_trackers["tracking_domain_id"])
)
tracking_domain_to_id = dict(
    zip(df_trackers["tracking_domain_id"], df_trackers["tracker_id"])
)
tracking_domain_set = set(df_trackers["tracking_domain_id"].values)
tracker_domain_lookup = np.array(
    [tracker_id_to_domain[i] for i in range(num_trackers)], dtype=np.int64
)

# ==============================================================================
# 2. LOAD TRAINING GRAPH & BUILD LEAKAGE-FREE TRAIN/VAL SPLIT
# ==============================================================================
df_tracking_train = pq.read_table("input/tracking_graph_train.parquet").to_pandas()
df_tracking_train["domain_id"] = df_tracking_train["domain_id"].astype(np.int64)
df_tracking_train["tracker_id"] = df_tracking_train["tracker_id"].astype(np.int16)
df_tracking_train["tracking_domain_id"] = df_tracking_train[
    "tracking_domain_id"
].astype(np.int64)

# Get all unique domains in training graph
all_known_domains = np.sort(df_tracking_train["domain_id"].unique())
num_known_domains = len(all_known_domains)

# Strictly isolate validation set: hold out 25,000 domains randomly
val_size = 25000
permuted_indices = np.random.RandomState(42).permutation(num_known_domains)
val_domain_ids = all_known_domains[permuted_indices[:val_size]]
train_domain_ids = all_known_domains[permuted_indices[val_size:]]

val_domain_set = set(val_domain_ids)
train_domain_set = set(train_domain_ids)

assert (
    len(train_domain_set.intersection(val_domain_set)) == 0
), "Train and validation overlap detected!"
assert (
    len(train_domain_set.intersection(test_domain_set)) == 0
), "Train and test overlap detected!"
assert (
    len(val_domain_set.intersection(test_domain_set)) == 0
), "Validation and test overlap detected!"

df_val_gt = df_tracking_train[
    df_tracking_train["domain_id"].isin(val_domain_set)
].copy()

# ==============================================================================
# 3. BUILD SPARSE TARGET MATRICES & TRACKER PRIORS (TRAIN ONLY)
# ==============================================================================
df_train_trackers = df_tracking_train[
    df_tracking_train["domain_id"].isin(train_domain_set)
].copy()

# Map domain_id to row index
train_id_to_idx = {d_id: idx for idx, d_id in enumerate(train_domain_ids)}
val_id_to_idx = {d_id: idx for idx, d_id in enumerate(val_domain_ids)}

# Build train CSR target matrix
train_rows = df_train_trackers["domain_id"].map(train_id_to_idx).values
train_cols = df_train_trackers["tracker_id"].values
train_data = np.ones(len(train_rows), dtype=np.float32)
y_train_csr = sparse.csr_matrix(
    (train_data, (train_rows, train_cols)),
    shape=(len(train_domain_ids), num_trackers),
    dtype=np.float32,
)

# Build validation CSR target matrix
val_rows = df_val_gt["domain_id"].map(val_id_to_idx).values
val_cols = df_val_gt["tracker_id"].values
val_data = np.ones(len(val_rows), dtype=np.float32)
y_val_csr = sparse.csr_matrix(
    (val_data, (val_rows, val_cols)),
    shape=(len(val_domain_ids), num_trackers),
    dtype=np.float32,
)

# Compute marginal tracker priors strictly on train split
train_tracker_counts = np.array(y_train_csr.sum(axis=0)).flatten()
tracker_priors = train_tracker_counts / len(train_domain_ids)

# Clean up memory
del (
    df_train_trackers,
    df_val_gt,
    train_rows,
    train_cols,
    train_data,
    val_rows,
    val_cols,
    val_data,
)
del df_tracking_train

# ==============================================================================
# 4. LOAD DOMAINS LOOKUP AND EXTRACT LEXICAL & MORPHOLOGY FEATURES
# ==============================================================================
all_target_domains = set(train_domain_ids).union(val_domain_set).union(test_domain_set)

domains_table = pq.read_table("input/domains.parquet", columns=["domain_id", "domain"])
df_domains_all = domains_table.to_pandas()
df_domains_all["domain_id"] = df_domains_all["domain_id"].astype(np.int64)

df_active_domains = df_domains_all[
    df_domains_all["domain_id"].isin(all_target_domains)
].copy()
del domains_table, df_domains_all

domain_id_to_name = dict(
    zip(df_active_domains["domain_id"], df_active_domains["domain"])
)
domain_name_to_id = {
    str(name).lower(): d_id
    for d_id, name in zip(df_active_domains["domain_id"], df_active_domains["domain"])
    if pd.notna(name)
}


def extract_lexical_features(domain_names):
    n = len(domain_names)
    char_len = np.zeros(n, dtype=np.float32)
    num_dots = np.zeros(n, dtype=np.float32)
    num_hyphens = np.zeros(n, dtype=np.float32)
    num_digits = np.zeros(n, dtype=np.float32)
    digit_ratio = np.zeros(n, dtype=np.float32)
    num_vowels = np.zeros(n, dtype=np.float32)
    vowel_ratio = np.zeros(n, dtype=np.float32)
    has_digits = np.zeros(n, dtype=np.float32)
    has_hyphens = np.zeros(n, dtype=np.float32)
    entropy = np.zeros(n, dtype=np.float32)
    tlds = []

    kw_shop = np.zeros(n, dtype=np.float32)
    kw_news = np.zeros(n, dtype=np.float32)
    kw_blog = np.zeros(n, dtype=np.float32)
    kw_tech = np.zeros(n, dtype=np.float32)
    kw_gov = np.zeros(n, dtype=np.float32)
    kw_edu = np.zeros(n, dtype=np.float32)
    kw_game = np.zeros(n, dtype=np.float32)
    kw_video = np.zeros(n, dtype=np.float32)
    kw_travel = np.zeros(n, dtype=np.float32)
    kw_food = np.zeros(n, dtype=np.float32)
    kw_auto = np.zeros(n, dtype=np.float32)
    kw_health = np.zeros(n, dtype=np.float32)

    vowel_set = set("aeiou")
    digit_set = set("0123456789")

    for i, name in enumerate(domain_names):
        if not isinstance(name, str) or len(name) == 0:
            tlds.append("unknown")
            continue

        l = len(name)
        char_len[i] = l
        name_lower = name.lower()

        dots = 0
        hyphens = 0
        digits = 0
        vowels = 0
        char_counts = {}

        for c in name_lower:
            char_counts[c] = char_counts.get(c, 0) + 1
            if c in digit_set:
                digits += 1
            elif c in vowel_set:
                vowels += 1
            elif c == ".":
                dots += 1
            elif c == "-":
                hyphens += 1

        num_dots[i] = dots
        num_hyphens[i] = hyphens
        has_hyphens[i] = 1.0 if hyphens > 0 else 0.0

        num_digits[i] = digits
        digit_ratio[i] = digits / l
        has_digits[i] = 1.0 if digits > 0 else 0.0

        num_vowels[i] = vowels
        vowel_ratio[i] = vowels / l

        # Shannon entropy
        ent = 0.0
        inv_l = 1.0 / l
        for cnt in char_counts.values():
            p = cnt * inv_l
            ent -= p * math.log2(p)
        entropy[i] = ent

        # TLD extraction
        if dots > 0:
            tld = name_lower.rsplit(".", 1)[-1]
        else:
            tld = "unknown"
        tlds.append(tld)

        kw_shop[i] = 1.0 if ("shop" in name_lower or "store" in name_lower or "cart" in name_lower or "buy" in name_lower or "pay" in name_lower) else 0.0
        kw_news[i] = 1.0 if ("news" in name_lower or "press" in name_lower or "media" in name_lower or "times" in name_lower or "post" in name_lower or "daily" in name_lower) else 0.0
        kw_blog[i] = 1.0 if ("blog" in name_lower or "forum" in name_lower or "wp" in name_lower or "community" in name_lower) else 0.0
        kw_tech[i] = 1.0 if ("tech" in name_lower or "dev" in name_lower or "api" in name_lower or "cloud" in name_lower or "git" in name_lower) else 0.0
        kw_gov[i] = 1.0 if "gov" in name_lower else 0.0
        kw_edu[i] = 1.0 if ("edu" in name_lower or "univ" in name_lower or "school" in name_lower or "academy" in name_lower) else 0.0
        kw_game[i] = 1.0 if ("game" in name_lower or "play" in name_lower or "bet" in name_lower or "casino" in name_lower) else 0.0
        kw_video[i] = 1.0 if ("video" in name_lower or "stream" in name_lower or "tv" in name_lower or "movie" in name_lower) else 0.0
        kw_travel[i] = 1.0 if ("travel" in name_lower or "hotel" in name_lower or "flight" in name_lower or "tour" in name_lower) else 0.0
        kw_food[i] = 1.0 if ("food" in name_lower or "recipe" in name_lower or "eat" in name_lower or "cook" in name_lower) else 0.0
        kw_auto[i] = 1.0 if ("auto" in name_lower or "car" in name_lower or "moto" in name_lower) else 0.0
        kw_health[i] = 1.0 if ("health" in name_lower or "med" in name_lower or "fit" in name_lower or "care" in name_lower) else 0.0

    features = {
        "char_len": char_len,
        "num_dots": num_dots,
        "num_hyphens": num_hyphens,
        "num_digits": num_digits,
        "digit_ratio": digit_ratio,
        "num_vowels": num_vowels,
        "vowel_ratio": vowel_ratio,
        "has_digits": has_digits,
        "has_hyphens": has_hyphens,
        "entropy": entropy,
        "kw_shop": kw_shop,
        "kw_news": kw_news,
        "kw_blog": kw_blog,
        "kw_tech": kw_tech,
        "kw_gov": kw_gov,
        "kw_edu": kw_edu,
        "kw_game": kw_game,
        "kw_video": kw_video,
        "kw_travel": kw_travel,
        "kw_food": kw_food,
        "kw_auto": kw_auto,
        "kw_health": kw_health,
        "tld": tlds,
    }
    return features


train_names = [domain_id_to_name.get(d, "") for d in train_domain_ids]
val_names = [domain_id_to_name.get(d, "") for d in val_domain_ids]
test_names = [domain_id_to_name.get(d, "") for d in test_domain_ids]

train_lex = extract_lexical_features(train_names)
val_lex = extract_lexical_features(val_names)
test_lex = extract_lexical_features(test_names)

# ==============================================================================
# 5. SUBWORD TF-IDF + SVD EMBEDDINGS (FIT STRICTLY ON TRAIN ONLY)
# ==============================================================================
sample_size = min(100000, len(train_names))
sample_train_names = [
    train_names[i]
    for i in np.random.RandomState(42).choice(
        len(train_names), sample_size, replace=False
    )
]

tfidf = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(3, 3), max_features=2500, min_df=5
)
tfidf.fit(sample_train_names)
sample_tfidf_matrix = tfidf.transform(sample_train_names)

n_svd = 16
svd = TruncatedSVD(n_components=n_svd, random_state=42)
svd.fit(sample_tfidf_matrix)
del sample_train_names, sample_tfidf_matrix


def transform_svd(names, tfidf, svd, batch_size=200000):
    n = len(names)
    results = np.zeros((n, n_svd), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_tfidf = tfidf.transform(names[start:end])
        results[start:end] = svd.transform(batch_tfidf).astype(np.float32)
    return results


train_svd = transform_svd(train_names, tfidf, svd)
val_svd = transform_svd(val_names, tfidf, svd)
test_svd = transform_svd(test_names, tfidf, svd)
del tfidf, svd

# ==============================================================================
# 6. GEOPOLITICAL & PRESS FREEDOM FEATURES
# ==============================================================================
try:
    df_fop = pd.read_csv("input/freedom-of-the-press.csv", sep="\t")
    if len(df_fop.columns) == 1:
        df_fop = pd.read_csv("input/freedom-of-the-press.csv", sep=",")
except Exception:
    df_fop = pd.read_csv("input/freedom-of-the-press.csv")

df_fop.columns = [c.strip().lower() for c in df_fop.columns]
tld_col = [c for c in df_fop.columns if "tld" in c][0]
score_col = [c for c in df_fop.columns if "freedom" in c][0]
df_fop["clean_tld"] = df_fop[tld_col].astype(str).str.lstrip(".").str.lower()
df_fop["score"] = pd.to_numeric(df_fop[score_col], errors="coerce")

tld_to_freedom = dict(zip(df_fop["clean_tld"], df_fop["score"]))
median_freedom = float(df_fop["score"].median())

ru_tlds = {"ru", "su", "рф", "by", "kz", "ua"}
eu_tlds = {
    "de",
    "uk",
    "fr",
    "nl",
    "it",
    "es",
    "pl",
    "cz",
    "se",
    "eu",
    "ch",
    "at",
    "be",
    "dk",
    "no",
    "fi",
}
asia_tlds = {"cn", "jp", "kr", "in", "vn", "tw", "id", "th"}
generic_tlds = {"com", "net", "org", "info", "biz"}
tech_tlds = {"io", "ai", "dev", "app", "tech", "cloud", "co"}


def get_geopolitical_features(tld_list):
    n = len(tld_list)
    freedom = np.zeros(n, dtype=np.float32)
    has_freedom = np.zeros(n, dtype=np.float32)
    is_ru = np.zeros(n, dtype=np.float32)
    is_eu = np.zeros(n, dtype=np.float32)
    is_asia = np.zeros(n, dtype=np.float32)
    is_generic = np.zeros(n, dtype=np.float32)
    is_tech = np.zeros(n, dtype=np.float32)

    for i, t in enumerate(tld_list):
        if t in tld_to_freedom:
            freedom[i] = tld_to_freedom[t]
            has_freedom[i] = 1.0
        else:
            freedom[i] = median_freedom
            has_freedom[i] = 0.0

        is_ru[i] = 1.0 if t in ru_tlds else 0.0
        is_eu[i] = 1.0 if t in eu_tlds else 0.0
        is_asia[i] = 1.0 if t in asia_tlds else 0.0
        is_generic[i] = 1.0 if t in generic_tlds else 0.0
        is_tech[i] = 1.0 if t in tech_tlds else 0.0

    return {
        "freedom_score": freedom,
        "has_freedom_score": has_freedom,
        "is_ru_region": is_ru,
        "is_eu_region": is_eu,
        "is_asia_region": is_asia,
        "is_generic_tld": is_generic,
        "is_tech_tld": is_tech,
    }


train_geo = get_geopolitical_features(train_lex["tld"])
val_geo = get_geopolitical_features(val_lex["tld"])
test_geo = get_geopolitical_features(test_lex["tld"])

# ==============================================================================
# 7. URL CLASSIFICATION TOPIC FEATURES
# ==============================================================================
df_urls = pd.read_csv("input/url-classification.csv", usecols=["url", "category"])
unique_cats = sorted(df_urls["category"].dropna().unique())
cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
num_cats = len(unique_cats)

url_records = []
urls_raw = df_urls["url"].astype(str).values
cats_raw = df_urls["category"].astype(str).values

for u, c in zip(urls_raw, cats_raw):
    if c not in cat_to_idx:
        continue
    if "://" in u:
        u = u.split("://", 1)[1]
    host = u.split("/", 1)[0].split(":", 1)[0].split("?", 1)[0].lower()
    if host.startswith("www."):
        host = host[4:]
    d_id = domain_name_to_id.get(host, None)
    if d_id is not None:
        url_records.append((d_id, cat_to_idx[c]))

del df_urls, urls_raw, cats_raw

domain_url_counts = {}
for d_id, c_idx in url_records:
    if d_id not in domain_url_counts:
        domain_url_counts[d_id] = np.zeros(num_cats, dtype=np.float32)
    domain_url_counts[d_id][c_idx] += 1.0
del url_records


def get_url_cat_features(domain_ids):
    n = len(domain_ids)
    cat_props = np.zeros((n, num_cats), dtype=np.float32)
    log_url_count = np.zeros(n, dtype=np.float32)
    has_url_cat = np.zeros(n, dtype=np.float32)

    for i, d in enumerate(domain_ids):
        if d in domain_url_counts:
            counts = domain_url_counts[d]
            total = counts.sum()
            if total > 0:
                cat_props[i] = counts / total
                log_url_count[i] = math.log1p(total)
                has_url_cat[i] = 1.0

    feat_dict = {
        f"url_cat_prop_{c}": cat_props[:, i] for i, c in enumerate(unique_cats)
    }
    feat_dict["log_url_count"] = log_url_count
    feat_dict["has_url_category"] = has_url_cat
    return feat_dict


train_url_feats = get_url_cat_features(train_domain_ids)
val_url_feats = get_url_cat_features(val_domain_ids)
test_url_feats = get_url_cat_features(test_domain_ids)
del domain_url_counts

# ==============================================================================
# 8. WEB GRAPH TOPOLOGY & TRACKER HYPERLINKS (STREAMING LINK GRAPH)
# ==============================================================================
max_domain_id = max(int(all_known_domains.max()), int(test_domain_ids.max()))
in_degree = np.zeros(max_domain_id + 1, dtype=np.int32)
out_degree = np.zeros(max_domain_id + 1, dtype=np.int32)

# Fast boolean lookup for active domains
is_active_target = np.zeros(max_domain_id + 1, dtype=bool)
is_active_target[all_known_domains] = True
is_active_target[test_domain_ids] = True

is_tracker_domain = np.zeros(max_domain_id + 1, dtype=bool)
valid_trackers = [d for d in tracking_domain_set if d <= max_domain_id]
is_tracker_domain[valid_trackers] = True

direct_tracker_counts = {}
top_tracker_links = {}

top_20_trackers = np.argsort(tracker_priors)[::-1][:20]
top_20_tracking_domains = {tracker_id_to_domain[t]: t for t in top_20_trackers}

link_parquet = pq.ParquetFile("input/link-graph.parquet")

for batch in link_parquet.iter_batches(
    batch_size=5000000, columns=["source_domain_id", "target_domain_id"]
):
    s_ids = batch["source_domain_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    t_ids = batch["target_domain_id"].to_numpy(zero_copy_only=False).astype(np.int64)

    mask_s = s_ids <= max_domain_id
    if np.any(mask_s):
        s_valid = s_ids[mask_s]
        bincount_s = np.bincount(s_valid)
        out_degree[: len(bincount_s)] += bincount_s.astype(np.int32)

    mask_t = t_ids <= max_domain_id
    if np.any(mask_t):
        t_valid = t_ids[mask_t]
        bincount_t = np.bincount(t_valid)
        in_degree[: len(bincount_t)] += bincount_t.astype(np.int32)

    is_to_tracker = np.zeros(len(t_ids), dtype=bool)
    if np.any(mask_t):
        is_to_tracker[mask_t] = is_tracker_domain[t_ids[mask_t]]

    if np.any(is_to_tracker):
        tr_sources = s_ids[is_to_tracker]
        tr_targets = t_ids[is_to_tracker]

        valid_src = tr_sources <= max_domain_id
        if np.any(valid_src):
            tr_sources = tr_sources[valid_src]
            tr_targets = tr_targets[valid_src]
            active_mask = is_active_target[tr_sources]
            if np.any(active_mask):
                for src, tgt in zip(tr_sources[active_mask], tr_targets[active_mask]):
                    direct_tracker_counts[src] = direct_tracker_counts.get(src, 0) + 1
                    if tgt in top_20_tracking_domains:
                        t_idx = top_20_tracking_domains[tgt]
                        if src not in top_tracker_links:
                            top_tracker_links[src] = set()
                        top_tracker_links[src].add(t_idx)


def get_graph_features(domain_ids):
    n = len(domain_ids)
    in_deg = np.zeros(n, dtype=np.float32)
    out_deg = np.zeros(n, dtype=np.float32)
    total_deg = np.zeros(n, dtype=np.float32)
    log_in_deg = np.zeros(n, dtype=np.float32)
    log_out_deg = np.zeros(n, dtype=np.float32)
    log_tot_deg = np.zeros(n, dtype=np.float32)
    deg_ratio = np.zeros(n, dtype=np.float32)
    direct_tr_cnt = np.zeros(n, dtype=np.float32)
    has_direct_tr = np.zeros(n, dtype=np.float32)

    top_tr_indicators = {
        f"links_to_top_tracker_{t}": np.zeros(n, dtype=np.float32)
        for t in top_20_trackers
    }

    for i, d in enumerate(domain_ids):
        ind = float(in_degree[d]) if d <= max_domain_id else 0.0
        outd = float(out_degree[d]) if d <= max_domain_id else 0.0
        tot = ind + outd

        in_deg[i] = ind
        out_deg[i] = outd
        total_deg[i] = tot
        log_in_deg[i] = math.log1p(ind)
        log_out_deg[i] = math.log1p(outd)
        log_tot_deg[i] = math.log1p(tot)
        deg_ratio[i] = (ind + 1.0) / (outd + 1.0)

        dtc = direct_tracker_counts.get(d, 0)
        direct_tr_cnt[i] = math.log1p(dtc)
        has_direct_tr[i] = 1.0 if dtc > 0 else 0.0

        if d in top_tracker_links:
            for t_idx in top_tracker_links[d]:
                top_tr_indicators[f"links_to_top_tracker_{t_idx}"][i] = 1.0

    feats = {
        "in_degree": in_deg,
        "out_degree": out_deg,
        "total_degree": total_deg,
        "log_in_degree": log_in_deg,
        "log_out_degree": log_out_deg,
        "log_total_degree": log_tot_deg,
        "degree_ratio": deg_ratio,
        "log_direct_tracker_links": direct_tr_cnt,
        "has_direct_tracker_links": has_direct_tr,
    }
    feats.update(top_tr_indicators)
    return feats


train_graph_feats = get_graph_features(train_domain_ids)
val_graph_feats = get_graph_features(val_domain_ids)
test_graph_feats = get_graph_features(test_domain_ids)
del in_degree, out_degree, is_active_target, direct_tracker_counts, top_tracker_links


# ==============================================================================
# 9. ASSEMBLE COMPREHENSIVE FEATURE MATRICES
# ==============================================================================
def assemble_dataframe(
    domain_ids, lex_feats, svd_feats, geo_feats, url_feats, graph_feats
):
    data = {"domain_id": domain_ids}
    for k, v in lex_feats.items():
        if k != "tld":
            data[k] = v
    for j in range(svd_feats.shape[1]):
        data[f"svd_{j}"] = svd_feats[:, j]
    data.update(geo_feats)
    data.update(url_feats)
    data.update(graph_feats)

    df = pd.DataFrame(data)
    feature_cols = [c for c in df.columns if c != "domain_id"]
    df[feature_cols] = df[feature_cols].fillna(0.0).astype(np.float32)
    return df


train_df = assemble_dataframe(
    train_domain_ids,
    train_lex,
    train_svd,
    train_geo,
    train_url_feats,
    train_graph_feats,
)
val_df = assemble_dataframe(
    val_domain_ids, val_lex, val_svd, val_geo, val_url_feats, val_graph_feats
)
test_df = assemble_dataframe(
    test_domain_ids, test_lex, test_svd, test_geo, test_url_feats, test_graph_feats
)

feature_columns = [c for c in train_df.columns if c != "domain_id"]

assert list(test_df["domain_id"]) == list(
    df_target["domain_id"]
), "Test domain_id order mismatch!"
assert (
    set(train_df.columns) == set(val_df.columns) == set(test_df.columns)
), "Feature schema mismatch across splits!"

# Clean intermediate dicts
del (
    train_lex,
    val_lex,
    test_lex,
    train_svd,
    val_svd,
    test_svd,
    train_geo,
    val_geo,
    test_geo,
)
del (
    train_url_feats,
    val_url_feats,
    test_url_feats,
    train_graph_feats,
    val_graph_feats,
    test_graph_feats,
)

# Compute marginal log-odds for prior head bias initialization
clipped_priors = np.clip(tracker_priors, 1e-5, 1.0 - 1e-5)
prior_logits = np.log(clipped_priors / (1.0 - clipped_priors)).astype(np.float32)


# ==============================================================================
# 10. MODEL ARCHITECTURE: TABULAR RESIDUAL NETWORK & ASYMMETRIC LOSS
# ==============================================================================
class TabularResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.drop1(self.act(self.fc1(self.norm1(x))))
        out = self.drop2(self.fc2(self.norm2(out)))
        return residual + out


class TrackerTabularResNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int = 355,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        dropout_rate: float = 0.2,
        initial_bias: np.ndarray = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        self.input_norm = nn.LayerNorm(in_features)
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.blocks = nn.ModuleList(
            [
                TabularResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

        self.tracker_interaction = nn.Linear(num_classes, num_classes, bias=False)
        nn.init.eye_(self.tracker_interaction.weight)

        if initial_bias is not None:
            self.head.bias.data.copy_(torch.from_numpy(initial_bias))
        else:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.input_norm(x)
        h = self.input_proj(x_norm)

        for block in self.blocks:
            h = block(h)

        h = self.final_norm(h)
        base_logits = self.head(h)
        interaction = self.tracker_interaction(base_logits)
        refined_logits = base_logits + 0.1 * interaction

        return refined_logits


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_neg: float = 2.0,
        gamma_pos: float = 0.0,
        clip_margin: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip_margin = clip_margin
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        targets = targets.float()

        pos_p = p.clamp(min=self.eps, max=1.0 - self.eps)
        pos_weight = (1.0 - pos_p) ** self.gamma_pos if self.gamma_pos > 0.0 else 1.0
        pos_loss = targets * pos_weight * torch.log(pos_p)

        neg_p = (p - self.clip_margin).clamp(min=0.0, max=1.0 - self.eps)
        neg_weight = (neg_p) ** self.gamma_neg
        neg_loss = (1.0 - targets) * neg_weight * torch.log(1.0 - neg_p + self.eps)

        loss = -torch.mean(pos_loss + neg_loss)
        return loss


in_features = len(feature_columns)
model = TrackerTabularResNet(
    in_features=in_features,
    num_classes=num_trackers,
    hidden_dim=256,
    num_blocks=2,
    dropout_rate=0.2,
    initial_bias=prior_logits,
).to(device)

criterion = AsymmetricLoss(gamma_neg=2.0, gamma_pos=0.0, clip_margin=0.05).to(device)
epochs = 5
optimizer = AdamW(
    model.parameters(), lr=2e-3, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8
)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

# ==============================================================================
# 11. CONSTRUCT TENSORS, DATALOADERS & EVALUATION METRIC
# ==============================================================================
X_train_tensor = torch.from_numpy(train_df[feature_columns].to_numpy(dtype=np.float32))
y_train_tensor = torch.from_numpy(y_train_csr.toarray().astype(np.float32))

X_val_tensor = torch.from_numpy(val_df[feature_columns].to_numpy(dtype=np.float32))
y_val_tensor = torch.from_numpy(y_val_csr.toarray().astype(np.float32))

X_test_tensor = torch.from_numpy(test_df[feature_columns].to_numpy(dtype=np.float32))

# Free large dataframes from memory
del train_df, val_df, test_df, y_train_csr, y_val_csr
gc.collect()

batch_size = 4096
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
    pin_memory=(device.type == "cuda"),
)

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size * 2,
    shuffle=False,
    drop_last=False,
    pin_memory=(device.type == "cuda"),
)


def evaluate_recall_at_10(model, data_loader, device):
    """Computes exact official competition metric: Recall@10 across all domains."""
    model.eval()
    total_recall = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            top10_preds = torch.topk(logits, k=10, dim=1).indices

            hits = torch.gather(batch_y, 1, top10_preds).sum(dim=1)
            true_counts = batch_y.sum(dim=1).clamp(min=1.0)
            recalls = hits / true_counts

            total_recall += recalls.sum().item()
            total_samples += batch_x.size(0)

    return total_recall / total_samples if total_samples > 0 else 0.0


# ==============================================================================
# 12. TRAINING LOOP WITH EARLY STOPPING
# ==============================================================================
best_val_recall = -1.0
best_model_state = None
patience = 4
epochs_no_improve = 0

for epoch in range(epochs):
    model.train()
    running_train_loss = 0.0
    num_train_batches = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_train_loss += loss.item()
        num_train_batches += 1

    scheduler.step()
    avg_train_loss = running_train_loss / max(1, num_train_batches)

    val_recall = evaluate_recall_at_10(model, val_loader, device)
    print(
        f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.5f} | Val Recall@10: {val_recall:.5f}"
    )

    if val_recall > best_val_recall:
        best_val_recall = val_recall
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            break

    # Hard time safety limit (3 hours max)
    if time.time() - script_start_time > 3.0 * 3600:
        print("Reached training time safety limit. Stopping training early.")
        break

# Restore best validation checkpoint
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    torch.save(best_model_state, "working/best_tracker_model.pt")

final_val_score = evaluate_recall_at_10(model, val_loader, device)

# ==============================================================================
# 13. TEST INFERENCE & FORMATTED SUBMISSION GENERATION
# ==============================================================================
model.eval()
test_top10_list = []
eval_batch_size = 4096

with torch.no_grad():
    for start_idx in range(0, len(X_test_tensor), eval_batch_size):
        end_idx = min(start_idx + eval_batch_size, len(X_test_tensor))
        batch_x = X_test_tensor[start_idx:end_idx].to(device)
        batch_logits = model(batch_x)
        batch_top10 = torch.topk(batch_logits, k=10, dim=1).indices.cpu().numpy()
        test_top10_list.append(batch_top10)

test_top10_tracker_ids = np.vstack(test_top10_list)
test_pred_tracking_domains = tracker_domain_lookup[test_top10_tracker_ids]

num_test_domains = len(test_domain_ids)
df_submission = pd.DataFrame(
    {
        "domain_id": np.repeat(test_domain_ids, 10),
        "tracking_domain_id": test_pred_tracking_domains.flatten(),
    }
)

submission_csv_path = "submission/submission.csv"
submission_tsv_path = "submission/submission.tsv"

df_submission.to_csv(submission_csv_path, sep="\t", index=False)
df_submission.to_csv(submission_tsv_path, sep="\t", index=False)

assert len(df_submission) == num_test_domains * 10, "Submission row count mismatch!"
assert set(df_submission.columns) == {
    "domain_id",
    "tracking_domain_id",
}, "Submission columns mismatch!"
assert (
    df_submission.isna().sum().sum() == 0
), "Submission contains unexpected null values!"

print(f"Final Validation Score: {final_val_score}")
