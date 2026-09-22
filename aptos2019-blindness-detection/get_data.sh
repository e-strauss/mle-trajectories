#!/usr/bin/env bash
# Kaggle APTOS 2019 Blindness Detection -- retina images, ~10 GB. Scored with
# quadratic-weighted Cohen kappa.
#
# Only train.csv and train_images/ are read by the pipelines here. The train
# side already exists on this machine at
# ~/repos/mle-star/machine_learning_engineering/tasks/img_classification_task1
# (train.csv + 3662 images, 8.1 GB); copying or symlinking that into ./input/
# covers every pipeline in this folder without the download.
#
# Populates ./input/, which is where every pipeline in this dataset's runs reads
# from ("./input/train.csv"), and what skrubify's --run-in expects to find:
#
#     python -m skrubify <run>/pipelines/x.py --run-in aptos2019-blindness-detection
#
# For a sample small enough to run inside the repair loop, see the
# dataset-sample skill; it writes ./sample/input/ from ./input/.
set -euo pipefail

SLUG=aptos2019-blindness-detection
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IN="$DIR/input"
EXPECTED=(train.csv test.csv train_images test_images sample_submission.csv)

have_all() {
    local f
    for f in "${EXPECTED[@]}"; do
        [ -e "$IN/$f" ] || return 1
    done
}

if [ "${1:-}" != "--force" ] && have_all; then
    echo "$IN already holds ${EXPECTED[*]} -- nothing to do (--force re-downloads)."
    exit 0
fi

command -v kaggle >/dev/null || {
    echo "kaggle CLI not on PATH. It is a declared project dependency, so" >&2
    echo "'uv sync' then '.venv/bin/kaggle' (or activate the venv) gets it." >&2
    exit 1
}
[ -f "${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json" ] || {
    echo "No kaggle.json. Create an API token at https://www.kaggle.com/settings" >&2
    echo "and save it as ~/.kaggle/kaggle.json (chmod 600)." >&2
    exit 1
}

mkdir -p "$IN"
# The competition rules must be accepted in the browser once. Until they are,
# this returns 403 rather than the data -- it is not a credentials problem.
kaggle competitions download -c "$SLUG" -p "$IN"
unzip -o -q "$IN/$SLUG.zip" -d "$IN"
rm "$IN/$SLUG.zip"
# The competition archive carries the images as train_images/<id>.png paths, so
# one unzip already yields the folders the pipelines glob over. This loop is a
# safety net for the nested-zip shape Kaggle also serves for some competitions;
# it is a no-op when there is nothing left to unpack.
for z in "$IN"/*.zip; do
    [ -e "$z" ] || break
    unzip -o -q "$z" -d "$IN"
    rm "$z"
done

missing=()
for f in "${EXPECTED[@]}"; do
    [ -e "$IN/$f" ] || missing+=("$f")
done
if [ ${#missing[@]} -gt 0 ]; then
    echo "! download finished but $IN is missing: ${missing[*]}" >&2
    exit 1
fi

echo "Done. $(du -sh "$IN" | cut -f1) in $IN:"
ls "$IN"
