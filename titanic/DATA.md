# titanic — where the data comes from

[Kaggle Titanic](https://www.kaggle.com/competitions/titanic). The competition's
public test file has no labels, so `./get_data.sh` downloads the competition
and splits its labeled `train.csv` (891 rows) into a new train and test. That
is what gives this task a test set that can be scored. Kaggle's own test file
is not used.

The recipe is `[prepare]` in `dataset.toml`: a stratified holdout on
`Survived`, `test_size` 0.2, seed 42.

Pipelines read `./input/`. `input/` is gitignored. The competition download
itself stays in `raw/`, which is also gitignored.

| file | rows | `Survived` |
| --- | --- | --- |
| `input/train.csv` | 712 | kept |
| `input/test.csv` | 179 | removed |

`scoring/answer.csv` is the removed `Survived` column, one value per test row,
in the same order. It is gitignored, so the agent never sees it.
