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

from ms_moe_maker import builder
from ms_moe_maker.recipe import parse


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """Run run_pipeline with every stage replaced by a recorder."""
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
    data_mod.generate_agent_traces = lambda config, callback=None: None

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

    # BOTH sys.modules AND the package attribute.
    #
    # `from . import router as router_mod` reads the ATTRIBUTE on the parent
    # package once that package has one - which it does as soon as any earlier
    # test imported the real module. Patching sys.modules alone therefore
    # worked when this file ran on its own and silently did nothing in a full
    # suite run, which is the worst way for a fixture to be wrong: green
    # in isolation, green-looking in aggregate, testing the real modules.
    import ms_moe_maker
    for name, mod in (("data", data_mod), ("finetune", ft),
                      ("stitch", stitch), ("router", router),
                      ("export", export)):
        monkeypatch.setitem(sys.modules, f"ms_moe_maker.{name}", mod)
        monkeypatch.setattr(ms_moe_maker, name, mod, raising=False)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    # preflight would want a real box; it is not what this file tests
    pf = types.ModuleType("preflight")
    pf.run = lambda config, recipe, **kw: types.SimpleNamespace(
        ok=True, warnings=[], failures=[], checks=[])
    pf.render = lambda p: []
    monkeypatch.setitem(sys.modules, "ms_moe_maker.preflight", pf)
    monkeypatch.setattr(ms_moe_maker, "preflight", pf, raising=False)

    rec, _ = parse({
        "schema_version": 1, "name": "t", "size": "0.5B",
        "corpus": {"min_samples": 1, "max_samples": 50},
        "experts": [
            {"name": "python", "source": {"kind": "stack", "language": "Python"}},
            {"name": "csharp", "source": {"kind": "stack", "language": "C#"}}]})

    monkeypatch.setattr("ms_moe_maker.config.resolve_roots",
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
    assert seen["finetune"] == corpora


def test_router_corpora_are_the_ones_the_experts_trained_on(wired):
    """The router must be mixed from what the experts actually saw. For a
    synth expert that is the generated trace file, not code_paths[name]."""
    _, seen, _, _ = wired
    assert seen["router_arg"] == seen["finetune"]


def test_stages_are_reported_done_not_skipped_on_a_fresh_run(wired):
    result, _, _, _ = wired
    assert "stitch" in result.stages_completed
    assert "router" in result.stages_completed


class TestRouterRefusesTheWrongShape:
    """Even if a future caller gets it wrong again, it fails in one second
    with the reason instead of ten minutes in with an errno."""

    def test_directories_are_refused_by_name(self, tmp_path):
        from ms_moe_maker.router import train_router
        d = tmp_path / "specialist_python"
        d.mkdir()
        cfg = types.SimpleNamespace(output_root=str(tmp_path), force=False)
        with pytest.raises(Exception) as exc:
            train_router(cfg, "moe", ["python"], {"python": str(d)})
        msg = str(exc.value)
        assert "specialist_dirs" in msg or "corpus" in msg.lower(), msg
        assert "IsADirectory" not in exc.typename, (
            "the wrong shape must be named, not discovered at open()")
