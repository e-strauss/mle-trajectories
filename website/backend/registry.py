"""The corpus registry: every dataset and agent run in the repo, and what state each is in.

A run's state is derived from its folder wherever possible, so a new skrubify
folder or runtime store shows up without anyone editing a manifest. Two small
TOML files record only what the folder cannot say:

* ``<dataset>/dataset.toml`` -- display name, task, metric, where the data lives.
* ``<dataset>/<run>/run.toml`` -- which agent produced the run, notes, and
  annotations on its pipeline sources (the main one, folders to leave out, ...).

A third, optional file holds what only the held-out labels can tell:

* ``<dataset>/<run>/test_scores.toml`` -- the run's score on the secret test
  set, computed outside the repo (the labels are never committed).

Everything a view or an action needs to decide whether it is available is in
:meth:`Run.facts`; nothing here imports skrub, stratum or a pipeline, so building
the registry takes well under a second and can be redone on every request.

    python -m website.backend.registry            # one line per run
    python -m website.backend.registry --run ttt-task/mlevolve_run_2
    python -m website.backend.registry --json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from dataset_sample._manifest import MANIFEST, fingerprint, is_sample

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTS = ("mle-star", "mlevolve", "mle-claude", "nano-mle", "aide")
# Agents whose pipelines are skrub plans already, with lineage in pipelines/results.json.
SKRUB_PLAN_AGENTS = ("mle-claude", "nano-mle")
TRAJECTORY_FILES = ("final_state.json", "journal_slim.json")

_DATASET_KEYS = {"label", "task", "data", "note", "metric", "defaults", "sample", "prepare"}
_DEFAULTS_KEYS = {"source", "runtime"}
_RUN_KEYS = {"agent", "label", "note", "trajectory", "originals", "metric", "sources",
             "runtime"}
_SOURCE_KEYS = {"label", "dirs", "default", "fold_identical_code", "hidden", "note"}
_RUNTIME_KEYS = {"label", "default", "hidden", "note"}
_METRIC_KEYS = {"name", "lower_is_better"}
TEST_SCORES_FILE = "test_scores.toml"
_TEST_KEYS = {"metric", "scored_at", "note", "final", "steps"}
_TEST_FINAL_KEYS = {"score", "pipeline", "file", "note"}


def _load_trajectory_module():
    """``pipeline_analyzer.trajectory`` by file: the package ``__init__`` pulls in
    stratum, and this module is stdlib only (``tools/trajectory.py`` does the same)."""
    path = REPO_ROOT / "tools" / "pipeline_analyzer" / "trajectory.py"
    spec = importlib.util.spec_from_file_location("_pa_trajectory", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod      # dataclasses resolve their module by name
    spec.loader.exec_module(mod)
    return mod


_traj = _load_trajectory_module()


def current_stratum_commit() -> str | None:
    """The installed stratum build, read the way ``pipeline_analyzer.runtime``
    stamps its stores (``stratum.__version__`` is static across builds)."""
    try:
        raw = importlib.metadata.distribution("stratum-ai").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    return ((json.loads(raw or "{}").get("vcs_info") or {}).get("commit_id"))


def _code_sha1(path: Path) -> str:
    # same key as pipeline_analyzer.runtime.code_sha1
    return hashlib.sha1(path.read_bytes()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class Metric:
    name: str | None = None
    lower_is_better: bool | None = None


@dataclass
class Source:
    """A set of folders holding pipeline files, looked up first-folder-wins."""
    name: str
    dirs: list[Path]
    skrub: bool                        # files are skrub DataOps plans
    label: str | None = None
    default: bool = False
    fold_identical_code: bool = False
    hidden: bool = False
    note: str | None = None
    files: dict[str, Path] = field(default_factory=dict)   # module stem -> file

    def rel_dirs(self, run_path: Path) -> list[str]:
        return [str(d.relative_to(run_path)) for d in self.dirs]


@dataclass
class StepInfo:
    module: str | None
    parent: str | None
    phase: str | None
    score: float | None
    desc: str | None = None
    test_score: float | None = None    # from test_scores.toml, filled in by load_run


@dataclass
class TestScores:
    """Scores on the held-out test labels (``test_scores.toml``), entered by hand
    or by a scoring script that reads the labels from outside the repo."""
    metric: str | None = None          # None: the run's own metric
    scored_at: str | None = None
    note: str | None = None
    final: float | None = None         # the submission the agent handed in
    final_pipeline: str | None = None  # the step that produced it
    final_file: str | None = None      # the predictions scored, relative to the run
    final_note: str | None = None
    steps: dict[str, float] = field(default_factory=dict)   # module -> test score


@dataclass
class RuntimeStore:
    path: Path
    source: str | None                 # name of the Source it measured
    data: str = "input"                # data folder it ran on: "input" or a sample folder
    legacy_rows: int | None = None     # measured with the removed --sample-rows cap
    n_ok: int = 0
    n_failed: int = 0
    n_code_changed: int = 0            # ok, but the pipeline file changed since
    n_old_build: int = 0               # ok, but measured under another stratum build
    n_data_changed: int = 0            # ok, but the sample was rebuilt differently since
    commits: list[str] = field(default_factory=list)   # stratum builds in the store
    measured_at: str | None = None     # newest entry
    label: str | None = None
    default: bool = False
    hidden: bool = False
    note: str | None = None

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def n_stale(self) -> int:
        return self.n_code_changed + self.n_old_build + self.n_data_changed

    @property
    def full_data(self) -> bool:
        return self.data == "input" and self.legacy_rows is None


@dataclass
class Dataset:
    name: str
    path: Path
    label: str
    task: str | None = None
    data: str = "local"                # "local" (./input) or a remote url
    note: str | None = None
    metric: Metric = field(default_factory=Metric)
    defaults: dict = field(default_factory=dict)   # [defaults] source / runtime
    sample_recipe: dict | None = None               # [sample]: how samples are built
    runs: list["Run"] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_input(self) -> bool:
        return (self.path / "input").is_dir()

    @property
    def samples(self) -> list[str]:
        """Folders holding a sampled ``input/`` (usable as ``--run-in``); hidden
        folders (a sample being built) are not samples yet."""
        return sorted(p.name for p in self.path.iterdir()
                      if p.is_dir() and p.name != "input" and not p.name.startswith(".")
                      and (p / "input").is_dir())

    def sample_info(self, name: str) -> dict:
        """What a sample folder's manifest says, in brief."""
        folder = self.path / name
        try:
            m = json.loads((folder / MANIFEST).read_text())
        except (OSError, json.JSONDecodeError):
            return {"name": name, "manifest": False}
        rows = {f: r for f, r in (m.get("rows") or {}).items() if isinstance(r, dict)}
        return {"name": name, "manifest": True, "size": m.get("size"),
                "built": m.get("built") or m.get("written"), "adopted": bool(m.get("adopted")),
                "recipe_sha1": m.get("recipe_sha1"), "note": m.get("note"),
                "rows": {f: [r.get("out"), r.get("src")] for f, r in rows.items()}}

    def data_status(self) -> str:
        if self.data != "local":
            return "remote"
        return "local" if self.has_input else "missing"


@dataclass
class Run:
    dataset: Dataset
    name: str
    path: Path
    agent: str | None
    label: str
    note: str | None = None
    metric: Metric = field(default_factory=Metric)
    trajectory_file: Path | None = None
    trajectory_meta: dict = field(default_factory=dict)
    originals: Source | None = None
    sources: list[Source] = field(default_factory=list)   # skrub plan sources
    runtime: list[RuntimeStore] = field(default_factory=list)
    steps: list[StepInfo] = field(default_factory=list)
    test: TestScores | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return f"{self.dataset.name}/{self.name}"

    @property
    def lineage_from(self) -> str | None:
        if self.trajectory_file:
            return "trajectory"
        if self.agent in SKRUB_PLAN_AGENTS and self.steps:
            return "results.json"
        return None

    @property
    def modules(self) -> list[str]:
        """The pipelines of this run, in trajectory / results order."""
        return [s.module for s in self.steps if s.module]

    # --- which source / runtime store a view uses --------------------------
    # Resolution order: the page's explicit choice, the only one there is, the
    # run.toml default, the dataset.toml default, then a built-in rule. The
    # returned reason is shown next to the picker.
    def pick_source(self, requested: str | None = None) -> tuple[Source | None, str]:
        if requested:
            chosen = next((s for s in self.sources if s.name == requested), None)
            if chosen:
                return chosen, "selected"
        visible = [s for s in self.sources if not s.hidden]
        if not visible:
            return None, "none available"
        if len(visible) == 1:
            return visible[0], "only one"
        chosen = next((s for s in visible if s.default), None)
        if chosen:
            return chosen, "run.toml"
        chosen = next((s for s in visible if s.name == self.dataset.defaults.get("source")), None)
        if chosen:
            return chosen, "dataset.toml"
        # built-in: the source covering the most pipelines, the agent's own
        # plans first on a tie, then by name
        best = max(visible, key=lambda s: (self.coverage(s)[0], s is self.originals))
        return best, "most pipelines covered"

    def default_source(self) -> Source | None:
        return self.pick_source()[0]

    def pick_runtime(self, requested: str | None = None,
                     source: Source | None = None) -> tuple[RuntimeStore | None, str]:
        if requested:
            chosen = next((r for r in self.runtime if r.name == requested), None)
            if chosen:
                return chosen, "selected"
        visible = [r for r in self.runtime if not r.hidden]
        if not visible:
            return None, "none available"
        if len(visible) == 1:
            return visible[0], "only one"
        chosen = next((r for r in visible if r.default), None)
        if chosen:
            return chosen, "run.toml"
        # prefer stores that measured the selected source
        source = source or self.default_source()
        same = [r for r in visible if source and r.source == source.name] or visible
        kind = self.dataset.defaults.get("runtime")
        if kind in ("full", "sample"):
            match = [r for r in same if r.full_data == (kind == "full")]
            if match:
                return max(match, key=lambda r: (r.n_ok, r.measured_at or "")), "dataset.toml"
        best = max(same, key=lambda r: (r.full_data, r.n_ok, r.measured_at or ""))
        return best, "full data preferred, then most pipelines measured"

    def coverage(self, source: Source) -> tuple[int, int]:
        """(pipelines with a file in ``source``, pipelines in the run)."""
        mods = self.modules or list(source.files)
        return sum(1 for m in mods if m in source.files), len(mods)

    def best(self) -> StepInfo | None:
        scored = [s for s in self.steps if s.score is not None]
        if not scored:
            return None
        pick = min if self.metric.lower_is_better else max
        return pick(scored, key=lambda s: s.score)

    def facts(self) -> dict:
        """What exists for this run -- the inputs to every availability check."""
        src = self.default_source()
        return {
            "agent": self.agent,
            "trajectory": self.trajectory_file is not None,
            "test_scores": self.test is not None,
            "lineage": self.lineage_from,
            "steps": len(self.steps),
            "originals": len(self.originals.files) if self.originals else 0,
            "skrub_sources": {s.name: self.coverage(s) for s in self.sources},
            "default_source": src.name if src else None,
            "runtime_stores": [r.path.name for r in self.runtime],
            "data": self.dataset.data_status(),
            "samples": self.dataset.samples,
        }


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read_toml(path: Path, allowed: set, warnings: list[str]) -> dict:
    if not path.is_file():
        return {}
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        warnings.append(f"{path.name}: not valid TOML ({exc})")
        return {}
    for key in sorted(set(data) - allowed):
        warnings.append(f"{path.name}: unknown key {key!r}")
    return data


def _metric(raw, where: str, warnings: list[str]) -> Metric:
    if not isinstance(raw, dict):
        return Metric()
    for key in sorted(set(raw) - _METRIC_KEYS):
        warnings.append(f"{where}: unknown metric key {key!r}")
    return Metric(raw.get("name"), raw.get("lower_is_better"))


def _merge_metric(*metrics: Metric) -> Metric:
    """First non-None value per field, in priority order."""
    return Metric(next((m.name for m in metrics if m.name is not None), None),
                  next((m.lower_is_better for m in metrics
                        if m.lower_is_better is not None), None))


def _py_subdirs(folder: Path) -> list[Path]:
    """``folder`` plus its immediate sub-folders that hold ``.py`` files
    (MLE-STAR's ``pipelines/1/``, ``ensemble/``)."""
    subs = [p for p in sorted(folder.iterdir())
            if p.is_dir() and not p.name.startswith((".", "_")) and any(p.glob("*.py"))]
    return [folder, *subs]


def _index_files(dirs: list[Path]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for d in dirs:
        for f in sorted(d.glob("*.py")):
            if not f.stem.startswith("_"):
                files.setdefault(f.stem, f)
    return files


def _resolve_dirs(run_path: Path, rel: list[str], where: str,
                  warnings: list[str]) -> list[Path]:
    out = []
    for r in rel:
        d = (run_path / r).resolve()
        if run_path not in (d, *d.parents):
            warnings.append(f"{where}: {r!r} is outside the run folder")
        elif not d.is_dir():
            warnings.append(f"{where}: no such folder {r!r}")
        else:
            out.append(d)
    return out


def _steps_from_results(path: Path, warnings: list[str]) -> tuple[list[StepInfo], Metric]:
    try:
        rows = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"{path.name}: unreadable ({exc})")
        return [], Metric()
    steps = [StepInfo(module=r["pipeline"], parent=r.get("parent"), phase=None,
                      score=r.get("score"), desc=r.get("description"))
             for r in rows if isinstance(r, dict) and r.get("pipeline")]
    names = {r.get("metric") for r in rows if isinstance(r, dict) and r.get("metric")}
    return steps, Metric(name=names.pop() if len(names) == 1 else None)


def _num(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _load_test_scores(run: Run) -> None:
    """``test_scores.toml``: the final submission's test score and, where the run
    kept predictions per step, each step's. Per-step scores land on the steps."""
    path = run.path / TEST_SCORES_FILE
    if not path.is_file():
        return
    w = run.warnings
    raw = _read_toml(path, _TEST_KEYS, w)
    final = raw.get("final") if isinstance(raw.get("final"), dict) else {}
    for key in sorted(set(final) - _TEST_FINAL_KEYS):
        w.append(f"{path.name} [final]: unknown key {key!r}")
    steps_raw = raw.get("steps") if isinstance(raw.get("steps"), dict) else {}
    steps = {}
    for name, v in steps_raw.items():
        if _num(v) is None:
            w.append(f"{path.name} [steps]: {name} is not a number")
        else:
            steps[name] = _num(v)
    t = TestScores(metric=raw.get("metric"),
                   scored_at=str(raw["scored_at"]) if raw.get("scored_at") else None,
                   note=raw.get("note"), final=_num(final.get("score")),
                   final_pipeline=final.get("pipeline"), final_file=final.get("file"),
                   final_note=final.get("note"), steps=steps)
    if final and t.final is None:
        w.append(f"{path.name} [final]: score missing or not a number")
    if t.final is None and t.final_pipeline in steps:
        t.final = steps[t.final_pipeline]
    elif t.final is not None and t.final_pipeline and t.final_pipeline not in steps:
        steps[t.final_pipeline] = t.final
    known = set(run.modules)
    if known:
        unknown = sorted(set(steps) - known)
        if unknown:
            w.append(f"{path.name}: {len(unknown)} pipeline(s) not in the lineage "
                     f"(e.g. {unknown[0]})")
    if t.final_file and not (run.path / t.final_file).resolve().is_relative_to(run.path.resolve()):
        w.append(f"{path.name} [final] file: {t.final_file!r} is outside the run folder")
    for s in run.steps:
        if s.module in steps:
            s.test_score = steps[s.module]
    run.test = t


def _load_runtime(run: Run, commit: str | None, annotations: dict) -> None:
    """Attach every ``runtime_stats_*.json`` beside the run, matched to the source
    it measured through the folders recorded in the store's own meta."""
    by_rel = {rel: s for s in [*run.sources, *([run.originals] if run.originals else [])]
              for rel in s.rel_dirs(run.path)}
    for path in sorted(run.path.glob("runtime_stats_*.json")):
        try:
            store = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            run.warnings.append(f"{path.name}: unreadable ({exc})")
            continue
        meta = store.get("meta") or {}
        # stores record absolute paths, possibly from another machine: match the
        # tail of the first measured folder against the run's source folders
        source = None
        for measured in meta.get("pipelines") or []:
            parts = Path(measured).parts
            for rel, src in by_rel.items():
                n = len(Path(rel).parts)
                if Path(*parts[-n:]) == Path(rel):
                    source = src
                    break
            if source:
                break
        if source is None:
            run.warnings.append(f"{path.name}: cannot tell which pipeline folder it measured")
        # which data: the run-in folder is the dataset's own (input/) or one of
        # its sample folders; anything else (an old scratch dir) counts as input/
        run_in = Path(meta.get("run_in") or "").name
        data = run_in if run_in in run.dataset.samples else "input"
        rs = RuntimeStore(path=path, source=source.name if source else None, data=data,
                          legacy_rows=meta.get("sample_rows"))
        if rs.legacy_rows:
            run.warnings.append(f"{path.name}: measured with the removed --sample-rows cap "
                                f"({rs.legacy_rows:,} rows); re-measure on a sample folder")
        folder = run.dataset.path / data
        current_fp = fingerprint(folder) if data != "input" and is_sample(folder) else None
        # older stores stamp the build only once, in meta, not per entry
        store_commit = (meta.get("versions") or {}).get("stratum_commit")
        commits, stamps = set(), []
        for name, entry in (store.get("pipelines") or {}).items():
            if entry.get("status") != "ok":
                rs.n_failed += 1
                continue
            rs.n_ok += 1
            if entry.get("measured_at"):
                stamps.append(str(entry["measured_at"]))
            built = entry.get("stratum_commit") or store_commit
            commits.add(built or "unknown")
            f = source.files.get(name) if source else None
            if f is None or entry.get("code_sha1") != _code_sha1(f):
                rs.n_code_changed += 1
            elif commit and built != commit:
                rs.n_old_build += 1
            elif current_fp and entry.get("data_fingerprint") != current_fp:
                rs.n_data_changed += 1
        rs.commits = sorted(commits)
        rs.measured_at = max(stamps, default=None)
        ann = annotations.get(path.stem) or {}
        for key in sorted(set(ann) - _RUNTIME_KEYS):
            run.warnings.append(f"run.toml [runtime.{path.stem}]: unknown key {key!r}")
        rs.label, rs.note = ann.get("label"), ann.get("note")
        rs.default, rs.hidden = bool(ann.get("default")), bool(ann.get("hidden"))
        run.runtime.append(rs)
    for name in sorted(set(annotations) - {r.name for r in run.runtime}):
        run.warnings.append(f"run.toml: [runtime.{name}] names no runtime_stats file")
    if sum(1 for r in run.runtime if r.default) > 1:
        run.warnings.append("run.toml: more than one runtime store marked default")


def load_run(dataset: Dataset, path: Path, commit: str | None = None) -> Run:
    warnings: list[str] = []
    cfg = _read_toml(path / "run.toml", _RUN_KEYS, warnings)
    if not (path / "run.toml").is_file():
        warnings.append("no run.toml")
    agent = cfg.get("agent")
    if agent is not None and agent not in AGENTS:
        warnings.append(f"run.toml: unknown agent {agent!r} (known: {', '.join(AGENTS)})")
    run = Run(dataset=dataset, name=path.name, path=path, agent=agent,
              label=cfg.get("label") or path.name, note=cfg.get("note"),
              warnings=warnings)

    # --- the agent's own scripts --------------------------------------------
    orig_rel = cfg.get("originals")
    if orig_rel is not None:
        orig_dirs = _resolve_dirs(path, list(orig_rel), "run.toml originals", warnings)
    else:
        orig_dirs = _py_subdirs(path / "pipelines") if (path / "pipelines").is_dir() else []
    if orig_dirs:
        run.originals = Source(name="pipelines", dirs=orig_dirs,
                               skrub=agent in SKRUB_PLAN_AGENTS,
                               label="agent's plans" if agent in SKRUB_PLAN_AGENTS else "originals",
                               files=_index_files(orig_dirs))

    # --- lineage: a trajectory, or mle-claude's results.json -----------------
    derived_metric = Metric()
    traj_rel = cfg.get("trajectory")
    traj_path = (path / traj_rel) if traj_rel else next(
        (path / f for f in TRAJECTORY_FILES if (path / f).is_file()), None)
    if traj_path is not None:
        try:
            t = _traj.parse(json.loads(traj_path.read_text()), path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            warnings.append(f"{traj_path.name}: cannot parse ({exc})")
        else:
            run.trajectory_file = traj_path
            run.trajectory_meta = t.meta
            run.steps = [StepInfo(s.module, s.parent, s.phase, s.score, s.desc)
                         for s in t.steps]
            derived_metric = Metric(lower_is_better=t.lower_is_better)
    elif agent in SKRUB_PLAN_AGENTS and run.originals:
        results = run.originals.dirs[0] / "results.json"
        if results.is_file():
            run.steps, derived_metric = _steps_from_results(results, warnings)
        else:
            warnings.append(f"{agent} run without pipelines/results.json: no lineage")
    run.metric = _merge_metric(_metric(cfg.get("metric"), "run.toml", warnings),
                               derived_metric, dataset.metric)

    # --- skrub plan sources ---------------------------------------------------
    annotations = cfg.get("sources") or {}
    if agent in SKRUB_PLAN_AGENTS and run.originals:
        run.sources.append(run.originals)
    for folder in sorted(p for p in path.glob("skrubify*") if p.is_dir()):
        ann = annotations.get(folder.name) or {}
        for key in sorted(set(ann) - _SOURCE_KEYS):
            warnings.append(f"run.toml [sources.{folder.name}]: unknown key {key!r}")
        dirs = (_resolve_dirs(path, list(ann["dirs"]), f"[sources.{folder.name}] dirs", warnings)
                if "dirs" in ann else _py_subdirs(folder))
        run.sources.append(Source(
            name=folder.name, dirs=dirs, skrub=True, label=ann.get("label"),
            default=bool(ann.get("default")),
            fold_identical_code=bool(ann.get("fold_identical_code")),
            hidden=bool(ann.get("hidden")), note=ann.get("note"),
            files=_index_files(dirs)))
    known = {s.name for s in run.sources}
    for name in sorted(set(annotations) - known):
        warnings.append(f"run.toml: [sources.{name}] names no skrubify folder")
    if sum(1 for s in run.sources if s.default) > 1:
        warnings.append("run.toml: more than one source marked default")

    if run.steps and run.originals:
        missing = [m for m in run.modules if m not in run.originals.files]
        if missing:
            warnings.append(f"{len(missing)} pipeline(s) named by the lineage have no "
                            f"original file (e.g. {missing[0]})")
    _load_test_scores(run)
    _load_runtime(run, commit, cfg.get("runtime") or {})
    return run


def load_dataset(path: Path, commit: str | None = None) -> Dataset:
    warnings: list[str] = []
    cfg = _read_toml(path / "dataset.toml", _DATASET_KEYS, warnings)
    if not (path / "dataset.toml").is_file():
        warnings.append("no dataset.toml")
    defaults = cfg.get("defaults") or {}
    for key in sorted(set(defaults) - _DEFAULTS_KEYS):
        warnings.append(f"dataset.toml [defaults]: unknown key {key!r}")
    if defaults.get("runtime") not in (None, "full", "sample"):
        warnings.append("dataset.toml [defaults] runtime: expected \"full\" or \"sample\"")
    ds = Dataset(name=path.name, path=path, label=cfg.get("label") or path.name,
                 task=cfg.get("task"), data=cfg.get("data") or "local",
                 note=cfg.get("note"),
                 metric=_metric(cfg.get("metric"), "dataset.toml", warnings),
                 defaults=defaults, warnings=warnings,
                 sample_recipe=cfg.get("sample") if isinstance(cfg.get("sample"), dict) else None)
    for child in sorted(path.iterdir()):
        if _is_run_dir(child):
            ds.runs.append(load_run(ds, child, commit))
    return ds


def _is_run_dir(p: Path) -> bool:
    return (p.is_dir() and not p.name.startswith(".")
            and ((p / "run.toml").is_file() or (p / "pipelines").is_dir()))


def load_corpus(root: Path = REPO_ROOT) -> list[Dataset]:
    """Every folder under ``root`` holding at least one run, as a Dataset."""
    commit = current_stratum_commit()
    out = []
    for p in sorted(root.iterdir()):
        if (p.is_dir() and not p.name.startswith(".")
                and ((p / "dataset.toml").is_file() or any(_is_run_dir(c) for c in p.iterdir()))):
            out.append(load_dataset(p, commit))
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _fmt_score(run: Run) -> str:
    b = run.best()
    return "—" if b is None else f"{b.score:.5g}"


def _fmt_test(run: Run) -> str:
    t = run.test
    if t is None:
        return "—"
    return "?" if t.final is None else f"{t.final:.5g}"


def _fmt_sources(run: Run) -> str:
    src = run.default_source()
    if src is None:
        return "—"
    have, total = run.coverage(src)
    extra = len([s for s in run.sources if s is not src and not s.hidden])
    return f"{src.name} {have}/{total}" + (f" (+{extra})" if extra else "")


def _fmt_runtime(run: Run) -> str:
    if not run.runtime:
        return "—"
    return ", ".join(f"{r.n_ok}" + (f"/{r.n_stale} stale" if r.n_stale else "")
                     + (f"@{r.data}" if r.data != "input" else "") for r in run.runtime)


def _print_table(corpus: list[Dataset]) -> None:
    rows = [("run", "agent", "steps", "best", "test", "lineage", "skrub plans", "runtime", "data", "!")]
    for ds in corpus:
        for r in ds.runs:
            rows.append((r.id, r.agent or "?", str(len(r.steps)), _fmt_score(r),
                         _fmt_test(r),
                         r.lineage_from or "—", _fmt_sources(r), _fmt_runtime(r),
                         ds.data_status(), str(len(r.warnings) + len(ds.warnings)) if
                         (r.warnings or ds.warnings) else ""))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for i, row in enumerate(rows):
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip())
        if i == 0:
            print("  ".join("-" * w for w in widths))
    notes = [(ds.name, w) for ds in corpus for w in ds.warnings]
    notes += [(r.id, w) for ds in corpus for r in ds.runs for w in r.warnings]
    if notes:
        print("\nwarnings:")
        for where, w in notes:
            print(f"  {where}: {w}")


def _print_run(run: Run) -> None:
    print(f"{run.id}  ({run.agent or 'agent?'})  {run.label}")
    if run.note:
        print(f"  note: {run.note}")
    m = run.metric
    direction = {True: "lower is better", False: "higher is better", None: "direction ?"}
    print(f"  metric: {m.name or '?'} ({direction[m.lower_is_better]})")
    print(f"  lineage: {run.lineage_from or 'none'}"
          + (f" from {run.trajectory_file.name}" if run.trajectory_file else "")
          + f", {len(run.steps)} steps, best {_fmt_score(run)}")
    if run.test:
        t = run.test
        print(f"  test: {_fmt_test(run)}"
              + (f" ({t.final_pipeline})" if t.final_pipeline else "")
              + (f", {len(t.steps)} step(s) scored" if t.steps else "")
              + (f", metric {t.metric}" if t.metric else "")
              + (f", scored {t.scored_at}" if t.scored_at else ""))
    if run.originals:
        print(f"  originals: {len(run.originals.files)} files in "
              f"{', '.join(run.originals.rel_dirs(run.path))}")
    for s in run.sources:
        have, total = run.coverage(s)
        flags = [f for f, on in (("default", s is run.default_source()),
                                 ("fold-identical-code", s.fold_identical_code),
                                 ("hidden", s.hidden)) if on]
        print(f"  source {s.name}: {have}/{total} pipelines, dirs "
              f"{', '.join(s.rel_dirs(run.path))}" + (f"  [{', '.join(flags)}]" if flags else ""))
        if s.note:
            print(f"      {s.note}")
    for r in run.runtime:
        print(f"  runtime {r.path.name}: source {r.source or '?'}, {r.n_ok} ok, "
              f"{r.n_failed} failed, {r.n_stale} stale, data={r.data}" + (f", legacy cap {r.legacy_rows}" if r.legacy_rows else ""))
    print(f"  data: {run.dataset.data_status()}"
          + (f", samples {', '.join(run.dataset.samples)}" if run.dataset.samples else ""))
    for w in run.dataset.warnings + run.warnings:
        print(f"  ! {w}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    ap.add_argument("--run", help="show one run in detail (<dataset>/<run>)")
    ap.add_argument("--json", action="store_true", help="dump every run's facts as json")
    args = ap.parse_args(argv)
    corpus = load_corpus(args.root.resolve())
    runs = {r.id: r for ds in corpus for r in ds.runs}
    if args.run:
        if args.run not in runs:
            ap.error(f"no run {args.run!r}; known: {', '.join(runs)}")
        _print_run(runs[args.run])
    elif args.json:
        print(json.dumps({rid: {**r.facts(), "warnings": r.warnings}
                          for rid, r in runs.items()}, indent=1))
    else:
        _print_table(corpus)
    return 0


if __name__ == "__main__":
    sys.exit(main())
