"""pipeline_24 -- TUNE the MLP: capacity (width x depth) at a proper epoch budget.

pipeline_23 hit 0.96041 with a deliberately small net (2x256) trained only 15
epochs -- almost certainly under-trained. This grids the two biggest levers,
width and depth, at 35 epochs (up from 15) with a bit more dropout for the larger
nets, on the champion features / MPS. One run via choose_from; harness records the
best + full grid in extra.grid.

  hidden   in {256, 512}
  n_layers in {2, 3}      (so (512, 3) is the 'big capacity' corner)
  fixed: max_epochs=35, dropout=0.25, lr=1e-3, batch=8192
"""
import skrub

from common import load_xy
from features import best_features
from nn import TorchMLP

X, y = load_xy(target="Cover_Type")
X = best_features(X.skb.drop(cols="Id"))
model = TorchMLP(
    hidden=skrub.choose_from([256, 512], name="hidden"),
    n_layers=skrub.choose_from([2, 3], name="n_layers"),
    dropout=0.25, lr=1e-3, max_epochs=35, batch_size=8192, device="mps",
)
pred = X.skb.apply(model, y=y)

DESCRIPTION = "TUNE MLP: hidden{256,512} x n_layers{2,3}, 35 epochs, champion features"
PARENT = "pipeline_23"
