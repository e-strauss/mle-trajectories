"""+ `nbr_frac`: P(tracker | a random tracked neighbour).

Every neighbourhood block so far sums tracker MENTIONS, so a neighbour that runs
50 trackers outvotes 25 neighbours that each run one. That is the wrong
aggregation for the question "what do the sites around me use", and it is the
likely reason the >100-neighbour regime has the worst recall of any bucket
(0.798 vs 0.890 overall, data_exploration_5): those histograms are dominated by
a few tracker-heavy hubs and flatten towards global popularity.

`nbr_frac` instead collapses direction and multiplicity to DISTINCT neighbours
and asks what fraction of them use each tracker. This is deliberately not
derivable from the existing blocks -- they are linear in mention counts, and
per-neighbour normalisation is not -- which is the property pipeline_09 showed
a new block needs in order to be worth anything.

Scored as a SIBLING of pipeline_10.
"""
from common import attach_scoring, load_xy
from featureset import build_pred, files

X, y = load_xy(blocks=files("nbr_frac"))
pred = attach_scoring(build_pred(X, y, "nbr_frac"))

DESCRIPTION = ("pipeline_10 + per-neighbour-normalised neighbourhood: fraction "
               "of DISTINCT tracked neighbours using each tracker")
PARENT = "pipeline_10"
