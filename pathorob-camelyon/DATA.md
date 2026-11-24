# PathoROB Camelyon image classification

Sources: [PathoROB Camelyon on Hugging Face](https://huggingface.co/datasets/bifold-pathomics/PathoROB-camelyon)
and [PathoROB](https://github.com/bifold-pathomics/PathoROB).

The prepared task is a fixed set of 22,402 RGB 256x256 patches: 13,200 training,
3,600 validation, 3,600 hidden known-center test and 2,002 hidden unseen-center
test images. RUMC/UMCU are the development centers; CWZ/RST/LPON are held out.
The split uses seed 42 and is slide-disjoint. Camelyon16 patient linkage is
unavailable, so full patient independence is not established. Source filenames
and metadata were replaced with opaque IDs and clean PNGs.

The primary test metric is balanced accuracy on the unseen centers at a fixed
threshold of 0.5. Models were selected using validation only; test scores were
computed after the runs. Each run's `test_scores.toml` contains final and
per-step scores; `evaluation.json` also includes known-center, per-center and
AUROC results. This is an adapted classification task, not PathoROB's APD
representation-robustness benchmark.

Data and preparation scripts live at `/data/estrauss-ldap/pathorob/camelyon`. The local `input` symlink points
to its `task/input` directory. Private labels stay outside this repository.
See `task_description.txt` and `DATA_DICTIONARY.md` for the input schema.

To reproduce the prepared task:

```bash
PYTHON=/home/estrauss-ldap/repos/MLEvolve/.venv/bin/python bash /data/estrauss-ldap/pathorob/camelyon/scripts/reproduce.sh
```

Original run logs, preparation records and the import script remain under that
same `/data` directory. Submission files are retained locally and gitignored.
