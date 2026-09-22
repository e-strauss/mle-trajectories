# beaver_enroll — where the data comes from

**BEAVER**, a data-discovery benchmark over MIT's institutional data warehouse.
Not a Kaggle competition, and **not present on this machine** — so there is no
`get_data.sh` here and no local fast path. The runs in this folder cannot be
executed against real data until the benchmark is obtained.

`mle_star_flash_run_1/final_state.json` records `task_name: beaver`,
`task_type: Data dicovery and classification` [sic], and carries the full task
description. `mle_star_flash_run_3/result.txt` shows the real paths the data sat
at when it was last run by hand:
`/Users/USER/Documents/UNI/SS26/BEAVER/table_splits/train/*.csv`.

## Expected layout

Pipelines read `./input/`, so `--run-in beaver_enroll` needs at least:

```
beaver_enroll/input/
    gold_enrollment_train.csv        the labelled target table
    gold_enrollment_test.csv
    sample_submission.csv
    course_summary.csv
    course_attributes.csv
    subject_summary.csv
    faculty_summary.csv
    instructor_attributes.csv
    terms.csv  offerings.csv  courses.csv  subjects.csv
    table_splits/train/*.csv         the warehouse dump the task discovers from
    table_splits/test/*.csv
    eval/gold_enrollment_train.csv
```

`table_splits/train/` is the "pile of data" the task is about: dozens of
warehouse tables (`ACADEMIC_TERMS_ALL.csv`, `COURSE_CATALOG_SUBJECT_OFFERED.csv`,
`FCLT_ROOMS.csv`, `OPA_PERSON_CURRENT.csv`, …), most of which are irrelevant.
Finding the few that matter is the task.

One schema detail that broke the last manual run: **`TERM_CODE` must stay a
string** (`2016FA`), not be coerced to an integer.

## Caveat on these trajectories

Per `mle_star_flash_run_3/result.txt`, the agent never proved its solution
against real data:

> The agent did not prove the solution works on BEAVER or on unseen test data.
> It proved the script runs without crashing when data is missing and
> substitutes 10-row dummy data with integer terms, then reports F1 = 1.0.

The repo README says the same — these four runs are kept as trajectories, not as
working solutions. A `--run-in` / `--compare-source` comparison against real data
would therefore be the *first* real test of them, and several are expected to
fail on the string `TERM_CODE` and on the `table_splits` layout.

The dummy-data fallback is also a hazard for the run loop: a pipeline that
silently substitutes 10 rows when a file is missing will report a score rather
than an error, so an empty or wrong `input/` looks like success. Gate or remove
that fallback before trusting any number from this folder.

## Sampling

Once the data exists, use the `dataset-sample` skill →
`beaver_enroll/sample/input/`. This is the star-schema case, with the extra
wrinkle that the sample must keep `table_splits/` intact as a *discovery*
problem: sample `gold_enrollment_train.csv`, semi-join the relevant warehouse
tables down to surviving keys, and keep the irrelevant tables present (shrunk,
not deleted) — deleting them turns a discovery task into a join task.
