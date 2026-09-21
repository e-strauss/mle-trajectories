"""Run a skrubified pipeline under stratum's scheduler and print op statistics.

    TTT_INPUT=ttt-task/input_1m python ttt-task/profile_pipeline.py \
        ttt-task/mlevolve_run_2/skrubify_manual/0029_*.py

TTT_INPUT selects the dataset (default: the full task data). TTT_EPOCHS shortens
the training loop, which is the only way to see the plan's own cost at small
sample sizes -- with 12 epochs the Predictor dominates everything else.
"""
import importlib.util
import sys
import time

import stratum as skrub

path = sys.argv[1] if len(sys.argv) > 1 else (
    "ttt-task/mlevolve_run_2/skrubify_manual/"
    "0029_e141d0f7f68a492faa18ce40dd7b5286.py"
)
spec = importlib.util.spec_from_file_location("pipeline", path)
pipeline = importlib.util.module_from_spec(spec)
sys.modules["pipeline"] = pipeline
spec.loader.exec_module(pipeline)

start = time.time()
with skrub.config(scheduler=True, stats=True, stats_top_k=20):
    search = pipeline.pred.skb.make_grid_search(
        n_jobs=1, fitted=True, refit=False, scoring=pipeline.RECALL_AT_10
    )
print(f"TOTAL {time.time() - start:.1f}s   score={search.results_['scores'][0]:.6f}")
