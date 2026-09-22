# cover_type_multi_table — where the data comes from

Not a Kaggle competition, so there is no `get_data.sh` here. The task is a
**synthesised relational split** of Forest CoverType: one base table plus 16
satellite tables, built so that blindly merging every CSV scores worse than a
pipeline that picks its joins and aggregates the multi-row tables first.

`mle_star_run_2/final_state.json` records `task_name: autofeat-covertype` and
carries the full table manifest quoted below. `mle_star_run_1` has no state
file; where the two runs' file names differ, run 2 is authoritative.

## Expected layout

All files sit flat in `./input/`, so `--run-in cover_type_multi_table` needs:

| file | rows | join key | rows per key |
|---|---|---|---|
| `forest_patches.csv` | 423,680 | `patch_id` | 1.00 — **base table**, holds `class` |
| `patch_measurements.csv` | 4,235,646 | `patch_id` | **10.00** |
| `plot_notes.csv` | 635,538 | `parcel_id` | **1.50** |
| `parcels.csv` | 423,680 | `patch_id` | 1.00 |
| `stands.csv` | 423,680 | `patch_id` | 1.00 |
| `survey_units.csv` | 423,680 | `patch_id` | 1.00 |
| `soil_registry.csv` | 423,680 | `patch_id` | 1.00 |
| `parcel_land_status.csv` | 423,680 | `parcel_id` | 1.00 |
| `parcel_soil_addendum.csv` | 423,680 | `parcel_id` | 1.00 |
| `parcel_soil_records.csv` | 423,680 | `parcel_id` | 1.00 |
| `stand_land_status.csv` | 423,680 | `stand_id` | 1.00 |
| `stand_soil_atlas.csv` | 423,680 | `stand_id` | 1.00 |
| `stand_soil_records.csv` | 423,680 | `stand_id` | 1.00 |
| `county_soil_atlas.csv` | 423,680 | `survey_unit_id` | 1.00 |
| `nrcs_soil_map.csv` | 423,680 | `survey_unit_id` | 1.00 |
| `sensor_calibration.csv` | 423,680 | `survey_unit_id` | 1.00 |
| `usfs_soil_survey.csv` | 423,680 | `survey_unit_id` | 1.00 |

Plus two metadata files: `connections.csv` (which column pairs *can* be joined)
and `tables.json` (each table's key column(s); a pair means several rows per
entity).

Task: predict the binary `class` in `forest_patches.csv` (1 = Spruce/Fir,
2 = Lodgepole Pine, exactly balanced at 211,840 each). Metric **roc_auc**, 3-fold
stratified CV, seed 42. No test set and no submission — report the mean CV score.

The design matrix scored must have **exactly 423,680 rows**; several tables hold
more than one row per key, so a plain `merge` multiplies rows and puts copies of
the same patch in both folds. `patch_id`, `survey_unit_id`, `parcel_id` and
`stand_id` are identifiers, not features.

## Upstream

Pristine tables: the `autofeat/covertype/` folder of
<https://zenodo.org/records/12755408/files/autofeat-data.tar>.

The relational rebuild is done by a script that lives in the sibling mle-star
repo, not here:

```bash
python ~/repos/mle-star/scripts/build_autofeat_covertype.py <pristine_covertype_dir>
```

It writes into
`~/repos/mle-star/machine_learning_engineering/tasks/autofeat-covertype/`, which
is **not present on this machine** — only the builder script is. Copy its output
into `cover_type_multi_table/input/`.

What that script changes, and why it matters for sampling (from its own
docstring): the stock task has all 13 tables 1:1, so a glob-merge reconstructs
the original flat table for free. The rebuild takes every continuous cartographic
variable out of its 1:1 table and re-expresses it as ~8-12 noisy repeated
observations per patch in one long table (`patch_measurements.csv`,
per-observation noise 1.0× the column's own sd), then adds distractor tables with
no signal. So a single raw observation is a poor feature and the per-patch mean
is a good one. `SEED = 20240826`.

The companion `~/repos/mle-star/scripts/write_covertype_description.py` generates
the task description.

## Sampling

Use the `dataset-sample` skill → `cover_type_multi_table/sample/input/`. This is
the star-schema case: sample `forest_patches.csv` (stratified on `class`), then
semi-join every satellite down to the surviving keys — following `patch_id` →
`parcel_id` / `stand_id` / `survey_unit_id` through `parcels.csv`, `stands.csv`
and `survey_units.csv` first, since those are what map the base key onto the
other three.

Two properties a sample must preserve or the task stops measuring anything:

- `patch_measurements.csv` must keep **all ~10 observations** of every surviving
  patch. Sampling its rows independently destroys the aggregate-then-join signal
  that is the whole point of the rebuild.
- the row-count assertion in the pipelines is `len(X) == 423680`. A sample
  changes that number, so a sampled run will trip it — expect to see that
  assertion fail and read it as "the sample works", not as a conversion defect.
