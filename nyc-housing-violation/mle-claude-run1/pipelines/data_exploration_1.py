"""Exploration 1: lake triage, part 1 -- schema/metadata profile of every table.

Fine-grained skrub DataOps evaluated once with .skb.eval(). Reads only Parquet
metadata (no rows): per table -> row count, year partitions, candidate key
columns (BBL / BIN / borough-block-lot / address / coordinates) and date columns.
Writes nothing.
"""
import re

import gcsfs
import pandas as pd
import pyarrow.parquet as pq
import skrub

TOKEN = "/home/estrauss-ldap/datasets/housing_violation_risk/nyc-lake-agent-key.json"
LAKE = "mle-nyc-lake/tasks/housing_violation_risk/v1/lake/full"

KEY_PATTERNS = {
    "bbl": r"(^|_)bbl$|base_bbl|mappluto_bbl",
    "bin": r"(^|_)bin(_|$)|^bin_number$",
    "boro_block_lot": r"^(boro|borough|boroid|borocode|boro_code)$|^block$|^lot$",
    "address": r"house_?num|street_?name|^address|incident_address",
    "coords": r"^(latitude|longitude|lat|lon|lng|x|y|xcoord|ycoord|x_coord|y_coord)",
}


def connect(token):
    return gcsfs.GCSFileSystem(token=token)


def list_files(fs, lake):
    """One row per parquet file: table, path, year partition (if any)."""
    files = [f for f in fs.find(lake) if f.endswith(".parquet")]
    return pd.DataFrame({
        "table": [f[len(lake) + 1:].split("/", 1)[0] for f in files], "path": files,
        "year": [int(m.group(1)) if (m := re.search(r"year=(\d+)", f)) else None
                 for f in files]})


def file_metadata(files, fs):
    """Footer read per file: row count + column names/types."""
    meta = [pq.ParquetFile(fs.open(p)).metadata for p in files["path"]]
    return files.assign(n_rows=[m.num_rows for m in meta],
                        schema=[[(f.name, str(f.type)) for f in m.schema.to_arrow_schema()]
                                for m in meta])


def key_kinds(schema):
    names = [n for n, _ in schema]
    return ",".join(k for k, pat in KEY_PATTERNS.items()
                    if any(re.search(pat, c) for c in names))


def date_columns(schema):
    return ",".join(n for n, t in schema
                    if t.startswith(("timestamp", "date"))
                    or re.search(r"date|_dt$|datetime", n))[:120]


def year_span(years):
    y = years.dropna()
    return f"{int(y.min())}-{int(y.max())}" if len(y) else ""


fs = skrub.as_data_op(TOKEN).skb.apply_func(connect)
files = skrub.deferred(list_files)(fs, LAKE)
meta = skrub.deferred(file_metadata)(files, fs)
per_table = meta.groupby("table").agg(n_files=("path", "size"), n_rows=("n_rows", "sum"),
                                      years=("year", year_span), schema=("schema", "first"))
report = per_table.assign(n_cols=per_table["schema"].map(len),
                          keys=per_table["schema"].map(key_kinds),
                          date_cols=per_table["schema"].map(date_columns)
                          ).drop(columns="schema").reset_index()

if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_colwidth", 120, "display.max_rows", 200)
    print(report.skb.eval().to_string(index=False))
