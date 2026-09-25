"""+ `tok_pop`: IDF-weighted P(tracker | hostname token).

pipeline_08 put the hostname in as raw character n-grams and lost 0.0033. But
the hostname clearly carries signal -- the TLD alone, expressed as a
target-informed profile, scores 0.795 standalone. The difference is the FORM:
`tld_pop` hands the model a tracker distribution, while StringEncoder handed it
256 SVD dimensions of surface text and asked it to discover the mapping from
337k rows.

So this is the same information as pipeline_08, in the form that worked: for
each alphabetic hostname token ("blogspot", "noticias", "shop"), the tracker
distribution of the out-of-sample domains containing it, averaged over the
domain's own tokens and weighted by IDF so a rare, specific token dominates a
generic one. It is also the only block that can help the 8.4% of domains with
no tracked neighbour at all beyond their TLD prior.

Leakage: profiles use ONLY tracked domains outside the frozen sample, so no
scored row's label contributes to its own feature.

Scored as a SIBLING of pipeline_10.
"""
from common import attach_scoring, load_xy
from featureset import build_pred, files

X, y = load_xy(blocks=files("tok_pop"))
pred = attach_scoring(build_pred(X, y, "tok_pop"))

DESCRIPTION = ("pipeline_10 + IDF-weighted P(tracker | hostname token) profiles "
               "fitted out-of-sample (the supervised form of pipeline_08)")
PARENT = "pipeline_10"
