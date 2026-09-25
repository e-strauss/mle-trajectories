"""+ `direct`: hyperlinks between the domain and each tracker's OWN domain.

The first block in this workspace built from data no earlier feature touched.
Every previous block describes a domain through OTHER domains' tracking
relations; this one reads the 2.55M link-graph edges that point straight AT the
355 tracker hostnames (and the 1.11M pointing back out of them).

A probe before building measured why it should matter: a direct hyperlink
between a domain and a tracker's hostname coincides with a true tracking edge
**50.4% of the time, against a 0.55% base rate** -- roughly a 90x lift. It is
sparse (present for 10.9% of train and 16.9% of target domains, covering 3.05%
of true edges), so it cannot carry the whole task, but where it fires it is
near-decisive, and it is the kind of evidence the neighbourhood blocks
structurally cannot see.

Coverage being HIGHER on target rows than train rows is the safe asymmetry: the
feature is more available at submission time than in training, not less (the
opposite of the nbr_2h leak).

Scored as a SIBLING of pipeline_10 (same net, same 5 seeds, one extra block), so
the delta against 0.89517 is this block's marginal value.
"""
from common import attach_scoring, load_xy
from featureset import build_pred, files

X, y = load_xy(blocks=files("direct"))
pred = attach_scoring(build_pred(X, y, "direct"))

DESCRIPTION = ("pipeline_10 + direct link-graph edges to/from the 355 tracker "
               "domains (50.4% precision vs 0.55% base rate)")
PARENT = "pipeline_10"
