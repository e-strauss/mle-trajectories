#!/usr/bin/env bash
# Kaggle Titanic. The competition's public test file has no labels, so this
# downloads the competition and splits its labeled train.csv the way the
# MLE-STAR run saw it. The split is [prepare] in dataset.toml (stratified
# holdout on Survived, test_size 0.2, seed 42).
#
# Writes:
#   raw/                  the competition download (not what pipelines read)
#   input/train.csv       labeled slice
#   input/test.csv        holdout with Survived removed
#   scoring/answer.csv    held-out Survived, one row per test row
#
# Pipelines read ./input/. Both input/ and scoring/answer.csv are gitignored.
#
#     python -m skrubify <run>/pipelines/x.py --run-in titanic
set -euo pipefail

SLUG=titanic
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"
RAW="$DIR/raw"
IN="$DIR/input"
EXPECTED=("$IN/train.csv" "$IN/test.csv" "$DIR/scoring/answer.csv")

have_all() {
    local f
    for f in "${EXPECTED[@]}"; do
        [ -e "$f" ] || return 1
    done
}

if [ "${1:-}" != "--force" ] && have_all; then
    echo "input/ and scoring/answer.csv already present -- nothing to do (--force rebuilds)."
    exit 0
fi

command -v kaggle >/dev/null || {
    echo "kaggle CLI not on PATH. It is a declared project dependency, so" >&2
    echo "'uv sync' then '.venv/bin/kaggle' (or activate the venv) gets it." >&2
    exit 1
}
if [ ! -f "${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json" ] \
    && [ ! -f "${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/access_token" ] \
    && [ -z "${KAGGLE_API_TOKEN:-}" ]; then
    echo "No Kaggle credentials. Create an API token at https://www.kaggle.com/settings" >&2
    echo "and save it as ~/.kaggle/kaggle.json (chmod 600), or set KAGGLE_API_TOKEN." >&2
    exit 1
fi

rm -rf "$RAW"
mkdir -p "$RAW"
# The competition rules must be accepted in the browser once. Until they are,
# this returns 403 rather than the data -- it is not a credentials problem.
kaggle competitions download -c "$SLUG" -p "$RAW"
unzip -o -q "$RAW/$SLUG.zip" -d "$RAW"
rm "$RAW/$SLUG.zip"

if [ -x "$ROOT/.venv/bin/python" ]; then
    PY="$ROOT/.venv/bin/python"
else
    PY=python3
fi
PYTHONPATH="$ROOT/tools${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m task_split "$DIR"

missing=()
for f in "${EXPECTED[@]}"; do
    [ -e "$f" ] || missing+=("$f")
done
if [ ${#missing[@]} -gt 0 ]; then
    echo "! prepare finished but missing: ${missing[*]}" >&2
    exit 1
fi

echo "Done. $(du -sh "$IN" | cut -f1) in $IN:"
ls "$IN"
