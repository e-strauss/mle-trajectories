#!/usr/bin/env bash
# Kaggle New York City Taxi Fare Prediction. train.csv is ~5.7 GB / ~55M rows,
# so expect a long download.
#
# A full copy already exists on this machine at ~/datasets/nyc-taxi (the path
# mlevolve_run_{2,3}/config.yaml point at, and its train.csv is byte-for-byte the
# size Kaggle serves: 5,697,178,298). Symlinking or copying that into ./input/
# skips the download entirely.
#
# The archive also carries GCP-Coupons-Instructions.rtf; it lands in input/ and
# is harmless.
#
# Populates ./input/, which is where every pipeline in this dataset's runs reads
# from ("./input/train.csv"), and what skrubify's --run-in expects to find:
#
#     python -m skrubify <run>/pipelines/x.py --run-in nyc_taxi_fare
#
# For a sample small enough to run inside the repair loop, see the
# dataset-sample skill; it writes ./sample/input/ from ./input/.
set -euo pipefail

SLUG=new-york-city-taxi-fare-prediction
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IN="$DIR/input"
EXPECTED=(train.csv test.csv sample_submission.csv)

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
