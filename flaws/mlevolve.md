# MLEvolve — observed flaws

Observed while running [MLEvolve](https://github.com/InternScience/MLEvolve) @ `9c5c8a3` on
`ttt-task` (TrackTheTrackers, Recall@10), Gemini 3.8 Flash, 2026-09-10/11.
Two runs: 15 steps (best 0.8464) and 50 steps (in progress).

Line references are to `9c5c8a3` unless noted.

---

## Harness bugs fixed before measuring

Both prevented the agent from producing *any* output; neither changes how it searches.

| bug | evidence | fix |
| --- | --- | --- |
| `max_output_tokens=16384` while `generate()` requests `thinking_level="high"`. Thinking tokens count against the cap, so code was truncated mid-statement, failed `is_valid_python_script`, and `extract_code` returned nothing. | Instrumented call: `finish_reason=MAX_TOKENS`, 9,513 thinking + 6,867 output tokens, **0 bytes** extractable. All 3 stepwise steps failed. | raised to 65536 (`llm/gemini.py:170,262`). After: `STOP`, 23,717 bytes of valid code, 0 extraction failures in 77 calls. |
| `base_url=""` passed straight to `genai.Client`, producing an **empty** base URL rather than the default endpoint. | `http_options={'base_url': ''}` → resolved base_url `''`; `None` → `https://generativelanguage.googleapis.com/`. | `or None` (`llm/gemini.py:131`), matching `llm/openai.py:113`. |

> Any run made with the stock 16 k cap and a high-thinking Gemini 3.x model is measuring a
> broken pipeline, not the agent.

---

## Flaws documented, not fixed

### Dead / silently overridden configuration

- **`num_gpus`** — declared (`config/__init__.py:62`), read nowhere. Setting it has no effect;
  GPU assignment comes only from `CUDA_VISIBLE_DEVICES` in the launcher.
- **`start_cpu_id`** — assigned (`engine/executor.py:69`), never read. Affinity slices from
  index 0 of the inherited mask, so two concurrent runs claim the same cores. `taskset` is the
  only way to partition.
- **`cpu_number`** — the value in `config.yaml` is silently overridden by the launcher's
  `cpu_number=` CLI arg via `OmegaConf.from_cli()` (`config/__init__.py:178`).
- **`STEP_LIMIT` / `TIME_LIMIT`** — exported by `run_single_task.sh`, consumed by no Python.

### Submission handling

- **`isolate_submission_path()` (`engine/executor.py:101`) matches only `.csv`.** It rewrites
  the generated source so parallel branches don't collide, but a solution writing
  `submission.tsv` escapes isolation entirely and every branch races on one shared file.
- **Worse, it gates the buggy flag.** `_determine_buggy` requires
  `submission/submission_<node_id>.csv` to exist and otherwise appends *"submission file not
  found"*; `_check_submission_file` only recovers a misplaced `.csv`, never a `.tsv`. A
  solution that correctly follows a TSV task spec is marked buggy despite succeeding.
  Our run survived only because the model hedged and wrote *both* names (both tab-separated).
- Consequence observed: the live `submission.tsv` was the last writer's (0.782 held-out),
  not the best node's (0.798).

### Selection signal — run-dependent, *not* a general flaw

Scoring every archived submission against held-out labels gave opposite answers in two runs,
so this is recorded as variance in the agent's self-validation, not a defect:

| | run 1 (15 steps) | run 2 (50 steps) |
| --- | --- | --- |
| nodes scored | 12 | 19 |
| Spearman(internal, held-out) | **+0.44** | **+0.986** |
| rank of true best, by internal | **7th of 12** | **1st** |
| cost of selecting on internal | 0.0142 recall | **0.0000** |
| mean optimism gap | +0.048 (top-6 cluster) | +0.0415 |

- In run 1 the top six internal scorers all overstated themselves by a near-identical
  **+0.047 … +0.052** while the rest had scattered *negative* gaps — one self-flattering
  validation lineage monopolised the leaderboard and buried the two genuinely best nodes
  (internally ranked 7th and 8th).
- In run 2 ranking was almost perfectly monotone (only trivial adjacent swaps), the agent's
  pick *was* the held-out best, and the optimism gap shrank monotonically with quality
  (+0.0778 at the worst node → +0.0298 at the best).

**Takeaway:** MLEvolve's internal validation can be either a near-perfect or a badly
misleading selector depending on which validation-split construction its lineage settles on,
and nothing in the search detects the difference. The risk is the variance, not a constant bias.

### Observability

- **No token or cost accounting.** `llm/__init__.py:64-78` unpacks `in_tok_count,
  out_tok_count` from both backends and returns only `output`. `generate()` — the streaming
  path used for all code generation — never reads `usage_metadata` at all.
- **No resource feedback.** The executor captures stdout, stderr and wall time only. Across 14
  nodes, exactly 1 received any memory-related feedback, and only because CUDA raised a hard
  exception. Host RSS, CPU time and GPU utilisation are never reported to the agent or the
  journal. (Measured post-hoc instead — see `tools/pipeline_analyzer`.)

### Multi-branch coordination

- **`parallel_search_num` is invisible to the agent.** It appears nowhere in
  `agents/prompts/`; the implementation guideline passes only time remaining, steps remaining
  and `exec_timeout` (`agents/prompts/impl_guideline.py:10-15`). With 3 branches each told it
  has a ~1 TB machine, the branches over-subscribe memory independently — observed 797 GB of
  1 TB used and swap exhausted, with single nodes at 409–433 GB RSS.

### Credentials

- **`agent.feedback.api_key` / `base_url` are ignored on the Gemini backend.**
  `_setup_gemini_client` is `@once` and builds the single client from `cfg.agent.code`
  (`llm/gemini.py:128-131`). Only the model name differs per stage.

---

## Behaviour notes (not defects)

- **Recovery drives progress.** In both runs the best solution came from a *debug* node
  repairing a failure: run 1's 0.8464 lineage began with `2860e90a` fixing a buggy parent;
  run 2's 0.8924 is `6730ca3b` repairing a timed-out `ab2208e1`.
- **Timeout feedback is good and the agent uses it — magnitude decides the outcome.** The
  diagnosis names the limit, the location and that no metric was produced. Of three repairs:
  epochs 20→8 + batch 512→2048 + SVD 32→16 → recovered at 0.8924; batch 512→1024 with epochs
  unchanged → timed out again; epochs 12→10 while dense `(N×355)` blocks went 5→6 → CUDA OOM
  allocating 153.92 GiB.
- **Scale is not the failure mode; representation is.** The 0.8924 node uses all 18.68 M
  candidate domains but stays sparse (CSR + `TruncatedSVD(16)`, hard-negative mining). The
  failures use the same domain count with dense domains×tracker blocks: 24.8 GB per block,
  ~301 GB peak across the `hstack`.
