# house_price — where the data comes from

Not a Kaggle competition, so there is no `get_data.sh` here. This file records
what `input/` has to contain and how it was built, because nothing else in the
repo does.

The runs point at it by absolute path: `mlevolve_run_{1,2}/config.yaml` has
`data_dir: /home/estrauss-ldap/datasets/house_price/input` and
`desc_file: .../house_price/task_description.txt`.

## Expected layout

Pipelines read `./input/` relative to their working directory, so
`--run-in house_price` needs:

```
house_price/input/
    train.parquet          22,114,250 rows, 1995-01-01 .. 2016-12-31, 175 MB
    test.csv                  375,098 rows, 2017-01-01 .. 2017-06-29, 24 MB
    sample_submission.csv     constant-price baseline, valid format
```

Held-out labels and the scorer are **not** under `input/` — they live beside it:

```
house_price/scoring/
    test_labels.csv        id,price for all 375,098 test rows
    score.py               RMSE on log10(price), needs only polars
```

Columns of `train.parquet` / `test.csv` (`price` present only in train):
`id, date, property_type, is_new_build, tenure, town, district, county,
sale_category`.

Task: predict `price`. Metric **RMSE on log10(price)**, lower better;
predictions are submitted in raw currency units and clipped to 1 before the log.
The split is **out-of-time**, not random — nothing in train postdates the test
period, and the price level roughly quadrupled over the training years, so the
level has to be extrapolated forward. Reference scores on the test set:
train-median constant 0.4175, best possible constant 0.3536, a
(`district`, `property_type`) log-mean lookup fitted on the last training year
0.2533.

## Upstream

The raw source is **HM Land Registry Price Paid Data** (UK, open licence):
<https://www.gov.uk/government/statistical-data-sets/price-paid-data>. The
complete single-file history is `pp-complete.csv`.

The copy this repo's runs were built from is a Kaggle mirror of the same data,
downloaded as `price_paid_records.csv` (2.4 GB), still present at
`~/datasets/house_price/raw/price_paid_records.csv`. Its header is the Land
Registry's own:

```
Transaction unique identifier,Price,Date of Transfer,Property Type,Old/New,
Duration,Town/City,District,County,PPDCategory Type,Record Status - monthly file only
```

The mirror's Kaggle slug was never recorded. Find it with
`kaggle datasets list -s "price paid"` before relying on it, or take the gov.uk
file, which is authoritative and needs no account.

## Derivation raw → `input/`

There is no script for this in the repo; it was done once, by hand, and the
result is what `~/datasets/house_price` holds. To reproduce it from
`price_paid_records.csv`:

1. Rename: `Price`→`price`, `Date of Transfer`→`date` (keep the date only, drop
   the `00:00` time), `Property Type`→`property_type`, `Old/New`→`is_new_build`,
   `Duration`→`tenure`, `Town/City`→`town`, `District`→`district`,
   `County`→`county`, `PPDCategory Type`→`sale_category`. Drop
   `Record Status - monthly file only`.
2. Replace the GUID transaction identifier with an opaque `id` of the form
   `R%08d`, assigned from a shuffle so it encodes neither date nor row order.
3. Split on `date`: `<= 2016-12-31` → `train.parquet` (with `price`);
   `2017-01-01 .. 2017-06-29` → `test.csv` (without `price`), and its labels →
   `scoring/test_labels.csv`.
4. `sample_submission.csv`: every test `id` with the constant training median
   (129,995).

Values are otherwise unfiltered administrative data — prices from 1 to
98,900,000, non-arm's-length sales (`sale_category` `B`), and location
boundaries that were redrawn during the period. Deciding what to clean is part
of the task, so **do not clean it during acquisition**.

## Fast path on this machine

```bash
mkdir -p house_price
ln -s ~/datasets/house_price/input   house_price/input
ln -s ~/datasets/house_price/scoring house_price/scoring
```

`pipeline_analyzer.runtime` checks for dangling symlinks under `input/` and
warns, so a broken link here is reported rather than silently failing at read
time (`tools/pipeline_analyzer/runtime.py:398-403`).

## Sampling

22M rows is far too much for skrubify's repair loop. Use the `dataset-sample`
skill to write `make_sample.py` → `house_price/sample/input/`. The one thing a
sample here must not do is **mix the two periods**: sample within
1995–2016 and within 2017 separately, keep the boundary intact, and keep the
last training year well represented, since that is where the extrapolation
signal is.
