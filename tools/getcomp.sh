#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <competition-name> [dir-name]" >&2
    exit 1
fi

COMP="$1"
DIR="${2:-$COMP}"

kaggle competitions download "$COMP"
unzip "${COMP}.zip" -d "$DIR"
rm "${COMP}.zip"

mkdir -p "$DIR/input"
mv "$DIR"/*.csv "$DIR/input/"

touch "$DIR/task_description.txt"

echo "Done. Files in $DIR/input:"
ls "$DIR/input"