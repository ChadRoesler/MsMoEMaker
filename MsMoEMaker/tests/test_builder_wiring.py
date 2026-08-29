"""What the orchestrator hands each stage.

Stage 5 of the first real 0.5B build died with

    IsADirectoryError: [Errno 21] Is a directory: '.../specialist_python'

because builder passed `specialist_dirs` (model checkpoints) into
train_router's `expert_paths` (corpus JSONL files). Both are Dict[str, str]
keyed by expert name, so nothing in the type system, the linter or the test
suite could tell them apart — and the mistake surfaced ten minutes into a run,
after two fine-tunes and a stitch.

Every stage here is faked, so this runs with no torch and in milliseconds. It
is not testing that training works; it is testing the PLUMBING — that the dict
of corpora goes where corpora are read and the dict of model dirs goes where
models are loaded. That is the failure this file exists for, and it is a whole
family rather than one bug.
"""
import json
import sys
import types

import pytest

from ms_moe_maker.run import builder
from ms_moe_maker.config.recipe import parse


def _base(path):
    """Strip the held-out-excluded suffix builder appends before fine-tuning.

    Specialists train on `<corpus>.train` so they never see the rows eval
    scores them against; `expert_corpus_paths` (and therefore train_router)
    keeps the BASE path, because train_router appends `.train` itself. That
    asymmetry is deliberate and is asserted below rather than papered over -
    if it ever drifts, `<corpus>.train.train` is the symptom.
    """
    return path[:-len(".train")] if path.endswith(".train") else path


@pytest.fixture
def wired(tmp_path, monkeypatch, request):
    """Run run_pipeline with every stage replaced by a recorder.

    Indirect-parametrise with a recipe dict to run a different shape
    through the same plumbing; without a param it is the two-stack-expert
    recipe every test below has always used.
    """
    monkeypatch.chdir(tmp_path)
    seen = {}

    data_root = tmp_path / "msmoe_data"
    out_root = tmp_path / "out"
    data_root.mkdir()
    out_root.mkdir()

    corpora = {}
    for name in ("python", "csharp"):
        p = data_root / f"{name}.jsonl"
        p.write_text("\n".join(
            json.dumps({"text": f"{name} body {i}"}) for i in range(20)),
            encoding="utf-8")
        corpora[name] = str(p)

    # ---- fake stage modules -------------------------------------------------
    data_mod = types.ModuleType("data")
    data_mod.collect_corpus = lambda config, languages, sources, callback: dict(corpora)
    # RECORDS, and takes **kw for the same reason verify_stitch's fake
    # does: a stub that mirrors today's signature exactly is a stub that
    # turns tomorrow's new argument into a TypeError in a fixture.
    def _fake_agent(config, callback=None, expert_name='agentcore', **kw):
        seen.setdefault('agent_calls', []).append(expert_name)
        p = data_root / f'{expert_name}_traces.jsonl'
        p.write_text('{"text": "trace"}\n', encoding='utf-8')
        return str(p)

    def _fake_reasoning(config, expert_name, callback=None, **kw):
        seen.setdefault('reasoning_calls', []).append(expert_name)
        p = data_root / f'{expert_name}_reasoning.jsonl'
        p.write_text('{"text": "think"}\n', encoding='utf-8')
        return str(p)

    data_mod.generate_agent_traces = _fake_agent
    data_mod.generate_reasoning_traces = _fake_reasoning

    # The third synth shape: a plain `kind: synth` expert (no reasoning, not
    # the tools expert) gets PLAIN DOMAIN TEXT, written to the same
    # `{expert}_code.jsonl` name the real generator uses.
    def _fake_domain(config, expert_name, callback=None, **kw):
        seen.setdefault('domain_calls', []).append(expert_name)
        p = data_root / f'{expert_name}_code.jsonl'
        p.write_text('{"text": "domain"}\n', encoding='utf-8')
        return str(p)

    data_mod.generate_domain_traces = _fake_domain

    ft = types.ModuleType("finetune")
    ft.specialist_is_done = lambda config, name: False

    def _fake_finetune(config, safe_name, data_path, expert_display=None):
        seen.setdefault("finetune", {})[safe_name] = data_path
        d = out_root / f"specialist_{safe_name}"
        d.mkdir(exist_ok=True)
        (d / "config.json").write_text("{}", encoding="utf-8")
        return str(d)

    ft.fine_tune_specialist = _fake_finetune

    stitch = types.ModuleType("stitch")
    stitch.stitch_is_done = lambda config: False
    stitch.stitch_moe = lambda config, names: str(out_root / "moe_untrained")
    # **kw so a new verify option does not break the fake. The fake's job
    # is to say 'the stitch was fine', not to mirror a signature.
    stitch.verify_stitch = lambda d, **kw: True

    router = types.ModuleType("router")
    router.router_is_done = lambda config: False

    def _fake_router(config, final_dir, safe_names, expert_corpus_paths):
        seen["router_arg"] = dict(expert_corpus_paths)
        return str(out_root / "moe_trained")

    router.train_router = _fake_router

    export = types.ModuleType("export")
    export.export_is_done = lambda config: False
    export.export_gguf = lambda config, final_dir: str(out_root / "m.gguf")

    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(
        is_available=lambda: False, empty_cache=lambda: None)

    # THE PACKAGE ATTRIBUTE, now that the modules live in subpackages.
    #
    # `from ..data import synth as data_mod` reads the ATTRIBUTE `synth` on
    # the data package - which the package's __init__ set the moment the real
    # module was imported. Patching sys.modules alone therefore worked when
    # this file ran on its own and silently did nothing in a full suite run,
    # which is the worst way for a fixture to be wrong: green in isolation,
    # green-looking in aggregate, testing the real modules. Patch the
    # attribute each import site actually reads.
    import ms_moe_maker.data as _data_pkg
    import ms_moe_maker.moe as _moe_pkg
    import ms_moe_maker.train as _train_pkg
    import ms_moe_maker.run as _run_pkg
    monkeypatch.setattr(_data_pkg, "synth", data_mod)
    monkeypatch.setattr(_train_pkg, "finetune", ft)
    monkeypatch.setattr(_train_pkg, "router", router)
    monkeypatch.setattr(_moe_pkg, "stitch", stitch)
    monkeypatch.setattr(_moe_pkg, "export", export)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    # preflight would want a real box; it is not what this file tests
    pf = types.ModuleType("preflight")
    pf.run = lambda config, recipe, **kw: types.SimpleNamespace(
        ok=True, warnings=[], failures=[], checks=[])
    pf.render = lambda p: []
    monkeypatch.setattr(_run_pkg, "preflight", pf)

    rec, _ = parse(getattr(request, "param", None) or {
        "schema_version": 1, "name": "t", "size": "0.5B",
        "corpus": {"min_samples": 1, "max_samples": 50},
        "experts": [
            {"name": "python", "source": {"kind": "stack", "language": "Python"}},
            {"name": "csharp", "source": {"kind": "stack", "language": "C#"}}]})

    monkeypatch.setattr("ms_moe_maker.config.pipeline.resolve_roots",
                        lambda size, dryrun: {"data": str(data_root),
                                              "output": str(out_root)})
    result = builder.run_pipeline(rec)
    return result, seen, corpora, out_root


def test_the_pipeline_completes(wired):
    result, _, _, _ = wired
    assert result.ok, result.message


def test_router_gets_corpora_not_model_directories(wired):
    """THE regression. Both are Dict[str, str] keyed by expert name."""
    _, seen, corpora, _ = wired
    arg = seen["router_arg"]
    assert arg == corpora, (
        f"train_router received {arg}, expected the corpus files {corpora}")
    for name, path in arg.items():
        assert path.endswith(".jsonl"), f"{name} -> {path} is not a corpus file"


def test_no_argument_to_router_is_a_directory(wired):
    import os
    _, seen, _, _ = wired
    for name, path in seen["router_arg"].items():
        assert not os.path.isdir(path), (
            f"{name} -> {path} is a directory; that is specialist_dirs "
            f"arriving where the corpora belong")


def test_finetune_gets_the_corpus_for_its_own_expert(wired):
    _, seen, corpora, _ = wired
    assert {k: _base(v) for k, v in seen["finetune"].items()} == corpora
    # ...and it is the SPLIT that arrives, not the whole corpus.
    for name, path in seen["finetune"].items():
        assert path.endswith(".train"), (
            f"{name} fine-tuned on {path!r} - the full corpus, including the "
            f"rows eval will score it against")


def test_specialists_never_see_the_held_out_rows(wired):
    """The regression test 0.4 did not have.

    train_router has had a fourteen-line comment about preferring `.train` so
    the router mix cannot eat the evaluation set. The specialists were handed
    the whole corpus and the split was not created until eval ran - so every
    expert trained on the rows it would later be scored against, and --mode
    quality read as "the MoE answers worse than one expert alone" when the gap
    was contamination.
    """
    import os
    _, seen, corpora, _ = wired
    for name, corpus in corpora.items():
        trained_on = seen["finetune"][name]
        assert trained_on != corpus, (
            f"{name} fine-tuned on the full corpus; eval holds out a slice of "
            f"this same file and would be scoring memorisation")
        held = corpus + ".heldout"
        assert os.path.exists(held), f"no held-out file written for {name}"
        with open(held, encoding="utf-8") as fh:
            held_rows = {ln.strip() for ln in fh if ln.strip()}
        with open(trained_on, encoding="utf-8") as fh:
            train_rows = {ln.strip() for ln in fh if ln.strip()}
        assert held_rows, f"held-out split for {name} is empty"
        assert not (held_rows & train_rows), (
            f"{len(held_rows & train_rows)} rows appear in BOTH {name}'s "
            f"training split and its held-out set")


def test_router_corpora_are_the_ones_the_experts_trained_on(wired):
    """The router must be mixed from what the experts actually saw. For a
    synth expert that is the generated trace file, not code_paths[name]."""
    _, seen, _, _ = wired
    # train_router resolves `.train` itself (same seed, same fraction), so it
    # is handed the base path and lands on the same rows the experts saw.
    assert seen["router_arg"] == {k: _base(v)
                                  for k, v in seen["finetune"].items()}


def test_stages_are_reported_done_not_skipped_on_a_fresh_run(wired):
    result, _, _, _ = wired
    assert "stitch" in result.stages_completed
    assert "router" in result.stages_completed


class TestRouterRefusesTheWrongShape:
    """Even if a future caller gets it wrong again, it fails in one second
    with the reason instead of ten minutes in with an errno."""

    def test_directories_are_refused_by_name(self, tmp_path):
        from ms_moe_maker.train.router import train_router
        d = tmp_path / "specialist_python"
        d.mkdir()
        cfg = types.SimpleNamespace(output_root=str(tmp_path), force=False)
        with pytest.raises(Exception) as exc:
            train_router(cfg, "moe", ["python"], {"python": str(d)})
        msg = str(exc.value)
        assert "specialist_dirs" in msg or "corpus" in msg.lower(), msg
        assert "IsADirectory" not in exc.typename, (
            "the wrong shape must be named, not discovered at open()")


# ── a GENERATED expert that is not the tools expert ─────────────────────────
#
# THE BUG THIS SECTION EXISTS FOR. builder computed `has_synth` across every
# expert and then called generate_agent_traces exactly once, always with
# expert_name=config.tools_expert_name. data.py deliberately skips kind=synth
# during corpus collection ("handled by generate_agent_traces"), so an expert
# named anything else got NOTHING — and the build died in the fine-tune loop
# with "No data path for expert X", hours in, after preflight and abliteration
# had already run and `build --plan` had reported `[ok] source/<name>
# kind=synth` on the way past.
#
# It survived because no test had a synth expert under a name other than the
# tools one. A path every real run goes around is untested by construction,
# and the missing test is the tell.

SYNTH_RECIPE = {
    "schema_version": 1, "name": "t", "size": "0.5B",
    "corpus": {"min_samples": 1, "max_samples": 50},
    "experts": [
        {"name": "python", "source": {"kind": "stack", "language": "Python"}},
        {"name": "csharp", "source": {"kind": "stack", "language": "C#"}},
        {"name": "shell", "source": {"kind": "synth"}},
    ],
}

REASONING_SYNTH_RECIPE = {
    "schema_version": 1, "name": "t", "size": "0.5B",
    "corpus": {"min_samples": 1, "max_samples": 50},
    "experts": [
        {"name": "python", "source": {"kind": "stack", "language": "Python"}},
        {"name": "csharp", "source": {"kind": "stack", "language": "C#"}},
        {"name": "shell", "source": {"kind": "synth", "reasoning": True}},
    ],
}


@pytest.mark.parametrize("wired", [SYNTH_RECIPE], indirect=True)
def test_a_named_synth_expert_gets_its_corpus_generated(wired):
    """THE regression. Without the fix the pipeline never completes.

    A plain synth expert now routes to generate_domain_traces (no think
    block, no tool calls) - the third synth shape."""
    result, seen, _, _ = wired
    assert result.ok, result.message
    assert "shell" in seen.get("domain_calls", []), (
        f"generate_domain_traces ran for {seen.get('domain_calls')}, "
        f"never for the synth expert 'shell'")
    assert "shell" not in seen.get("agent_calls", []), (
        "a plain synth expert must not get tool traces - that output is "
        "discarded, the domain corpus is what it fine-tunes on")


@pytest.mark.parametrize("wired", [SYNTH_RECIPE], indirect=True)
def test_the_synth_expert_finetunes_on_what_was_generated_for_it(wired):
    _, seen, _, _ = wired
    path = _base(seen["finetune"].get("shell") or "")
    assert path and path.endswith("shell_code.jsonl"), (
        f"shell fine-tuned on {path!r}; it must be its own generated corpus")


@pytest.mark.parametrize("wired", [SYNTH_RECIPE], indirect=True)
def test_the_router_is_mixed_from_the_synth_corpus_too(wired):
    _, seen, _, _ = wired
    assert seen["router_arg"] == {k: _base(v)
                                  for k, v in seen["finetune"].items()}, (
        "the router must be mixed from exactly what the experts trained on")


@pytest.mark.parametrize("wired", [REASONING_SYNTH_RECIPE], indirect=True)
def test_a_reasoning_synth_expert_is_not_also_given_tool_traces(wired):
    """Both generators can claim the same expert, but only one output is ever
    read: reasoning_paths wins in the fine-tune loop, so generating tool traces
    as well would be teacher hours spent on a file nothing opens."""
    result, seen, _, _ = wired
    assert result.ok, result.message
    assert "shell" in seen.get("reasoning_calls", [])
    assert "shell" not in seen.get("agent_calls", []), (
        "tool traces were generated for an expert whose corpus comes from "
        "the reasoning path — that output is discarded")
    assert _base(seen["finetune"]["shell"]).endswith("shell_reasoning.jsonl")
