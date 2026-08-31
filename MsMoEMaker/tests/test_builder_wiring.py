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


# The default shape every test here has always used. Hoisted out of the
# fixture so a test can re-run the pipeline with different arguments against
# the same fakes - which is how --only is checked below.
TWO_EXPERTS = {
    "schema_version": 1, "name": "t", "size": "0.5B",
    "corpus": {"min_samples": 1, "max_samples": 50},
    "experts": [
        {"name": "python", "source": {"kind": "stack", "language": "Python"}},
        {"name": "csharp", "source": {"kind": "stack", "language": "C#"}}]}


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
    data_mod.collect_corpus = lambda *a, **kw: dict(corpora)
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
        # WHAT it was handed, not just THAT it was called: `domains` is the
        # difference between a reasoning specialist that spans the roster and
        # one that is a second expert on somebody else's subject.
        seen.setdefault('reasoning_domains', {})[expert_name] = kw.get('domains')
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

    # RECORDS `retrain`, because that is the whole --only contract: the named
    # expert is treated as forced and every other one keeps its own answer.
    # A stub that swallowed the kwarg would let the flag stop arriving without
    # a single test noticing.
    def _fake_is_done(config, name, retrain=False):
        seen.setdefault("retrain", {})[name] = retrain
        return False

    ft.specialist_is_done = _fake_is_done

    def _fake_finetune(config, safe_name, data_path, expert_display=None,
                       **kw):
        seen.setdefault("finetune_retrain", {})[safe_name] = kw.get("retrain")
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
    # is to say 'the stitch was fine', not to mirror a signature - but it
    # RECORDS what it was handed, so the moe.* recipe knobs can be proven to
    # arrive (gate_fill and router_init_std used to be hardcoded 0.02 here
    # while the recipe fields sat dead).
    def _fake_verify(d, **kw):
        seen["verify_kwargs"] = dict(kw)
        return True

    stitch.verify_stitch = _fake_verify

    router = types.ModuleType("router")
    router.router_is_done = lambda config: False
    # builder asks for this to delete a router trained on a PREVIOUS skeleton
    # after a fresh stitch. Point it somewhere that does not exist, so the
    # rmtree branch is inert here - this fixture is testing plumbing, and a
    # stub that deletes real directories is a fixture with teeth.
    router.router_dir = lambda config: str(out_root / "moe_trained_absent")

    def _fake_router(config, final_dir, safe_names, expert_corpus_paths):
        seen["router_arg"] = dict(expert_corpus_paths)
        return str(out_root / "moe_trained")

    router.train_router = _fake_router

    export = types.ModuleType("export")
    export.export_is_done = lambda config: False
    export.export_gguf = lambda config, final_dir: str(out_root / "m.gguf")
    # The builder asks where the GGUF WOULD be so it can throw away one that
    # belongs to a router it just replaced. Real path, so the test below can
    # put a stale file there and watch it go.
    export.gguf_path_for = lambda config: str(out_root / "m.gguf")

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

    rec, _ = parse(getattr(request, "param", None) or TWO_EXPERTS)

    monkeypatch.setattr("ms_moe_maker.config.pipeline.resolve_roots",
                        lambda size, dryrun, *a, **kw: {
                            "data": str(data_root),
                            "output": str(out_root)})
    result = builder.run_pipeline(rec)
    return result, seen, corpora, out_root


def test_the_pipeline_completes(wired):
    result, _, _, _ = wired
    assert result.ok, result.message


class TestOnlyOneExpert:
    """`--only <expert>`: the README's "retrain one and re-splice" as a flag.

    Before it, a user had two options: --force (retrain all N experts) or
    delete one specialist by hand - which retrained it for hours and then had
    the stitch skip, because the stitch only compared expert names. The
    retrain was thrown away and the previous model was exported.

    These run against the same fakes as the fixture, so they are testing the
    PLUMBING: that the flag reaches the one predicate that decides a skip.
    """

    def test_only_forces_exactly_the_named_expert(self, wired):
        _, seen, _, _ = wired
        seen.clear()
        rec, _w = parse(TWO_EXPERTS)
        result = builder.run_pipeline(rec, only=["python"])
        assert result.ok, result.message
        assert seen["retrain"] == {"python": True, "csharp": False}
        assert seen["finetune_retrain"] == {"python": True, "csharp": False}, (
            "the skip predicate and the trainer must agree - a retrain that "
            "gets past the gate and then resumes a checkpoint is the same "
            "wasted build in a different place")

    def test_no_only_forces_nothing(self, wired):
        _, seen, _, _ = wired
        assert seen["retrain"] == {"python": False, "csharp": False}

    def test_a_typo_fails_before_anything_runs(self, wired):
        """`--only shel` must not become a no-op that reports success."""
        _, seen, _, _ = wired
        seen.clear()
        rec, _w = parse(TWO_EXPERTS)
        result = builder.run_pipeline(rec, only=["shel"])
        assert result.ok is False
        assert "shel" in result.message
        assert "python" in result.message and "csharp" in result.message, (
            "the message has to carry the real names - the reader is one "
            "keystroke away from the right command")
        assert "finetune" not in seen, "nothing may run on a bad name"

    def test_comma_and_repeat_are_both_accepted(self):
        assert builder.resolve_only(["a,b", "c"], ["a", "b", "c"]) == (
            ("a", "b", "c"), ())

    def test_case_is_forgiven_and_canonicalised(self):
        assert builder.resolve_only(["Python"], ["python"]) == (("python",), ())

    def test_duplicates_collapse(self):
        assert builder.resolve_only(["a", "a,b"], ["a", "b"]) == (
            ("a", "b"), ())

    def test_nothing_asked_is_nothing_forced(self):
        assert builder.resolve_only(None, ["a"]) == ((), ())
        assert builder.resolve_only([], ["a"]) == ((), ())


def test_a_new_router_discards_the_gguf_it_did_not_produce(wired):
    """The last link in the invalidation chain.

    A restitch already discarded the trained router. Nothing discarded the
    GGUF - export_is_done() only asks whether a .gguf and its .smokepass.txt
    exist, and both survive a retrain. So `--only shell` would restitch,
    retrain the router, print "[skip] GGUF export already done" and hand back
    the PREVIOUS model as the artifact of this build.
    """
    _, _, _, out_root = wired
    gguf = out_root / "m.gguf"
    proof = out_root / "m.gguf.smokepass.txt"
    gguf.write_bytes(b"the previous router's model")
    proof.write_text("smoke passed, for other bytes", encoding="utf-8")

    rec, _w = parse(TWO_EXPERTS)
    assert builder.run_pipeline(rec).ok
    assert not gguf.exists(), "a GGUF from the old router must not survive"
    assert not proof.exists(), (
        "a smoke pass belongs to the bytes it was run against; keeping it "
        "would let the next resume call this build done")


def test_verify_stitch_gets_the_recipe_moe_values(wired):
    """moe.shared_expert_gate_fill and moe.router_init_std must arrive at the
    verifier - they were hardcoded 0.02 there while the recipe fields sat
    dead, so the fill and its check were wrong together and agreed."""
    _, seen, _, _ = wired
    assert seen["verify_kwargs"].get("gate_fill") == 0.02
    assert seen["verify_kwargs"].get("router_init_std") == 0.02


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


@pytest.mark.parametrize("wired", [REASONING_SYNTH_RECIPE], indirect=True)
def test_a_hand_written_reasoning_expert_still_gets_one_domain(wired):
    """`reasoning: true` written by hand on a domain expert keeps EXACTLY
    today's behaviour. It was written to be a domain expert that thinks;
    spanning its corpus across the roster underneath a recipe nobody edited
    would be a silent corpus change."""
    _, seen, _, _ = wired
    assert seen["reasoning_domains"]["shell"] is None


# The injected roster expert: `reasoning_expert:` is a flag, not an entry in
# `experts:`, and it must arrive at the generator carrying the OTHER experts'
# domains - itself and the tools expert excluded.
ROSTER_REASONING_RECIPE = {
    "schema_version": 1, "name": "t", "size": "0.5B",
    "corpus": {"min_samples": 1, "max_samples": 50},
    "reasoning_expert": True,
    "tools_expert": True,
    "experts": [
        {"name": "python", "source": {"kind": "stack", "language": "Python"}},
        {"name": "csharp", "source": {"kind": "stack", "language": "C#"}},
    ],
}


@pytest.mark.parametrize("wired", [ROSTER_REASONING_RECIPE], indirect=True)
def test_the_injected_reasoning_expert_is_handed_the_roster(wired):
    result, seen, _, _ = wired
    assert result.ok, result.message
    assert "deliberation" in seen.get("reasoning_calls", []), (
        f"the injected reasoning expert never reached the generator: "
        f"{seen.get('reasoning_calls')}")
    domains = seen["reasoning_domains"]["deliberation"]
    assert domains == ["Python", "C#"], (
        f"roster domains were {domains!r}; it must be the OTHER experts' "
        f"display names, with itself and the tools expert excluded")


@pytest.mark.parametrize("wired", [ROSTER_REASONING_RECIPE], indirect=True)
def test_the_roster_expert_trains_on_its_own_reasoning_corpus(wired):
    _, seen, _, _ = wired
    assert _base(seen["finetune"]["deliberation"]).endswith(
        "deliberation_reasoning.jsonl")


# A roster with nothing on it: the only two experts are the two INJECTED ones,
# so after excluding itself and the tools expert there is no domain left to
# draw from. Legal, and it has to degrade to the single-domain corpus rather
# than hand the generator an empty list (rnd.choice([]) is an IndexError two
# hours into a build).
EMPTY_ROSTER_RECIPE = {
    "schema_version": 1, "name": "t", "size": "0.5B",
    "corpus": {"min_samples": 1, "max_samples": 50},
    "reasoning_expert": True,
    "tools_expert": True,
    "experts": [],
}


@pytest.mark.parametrize("wired", [EMPTY_ROSTER_RECIPE], indirect=True)
def test_an_empty_roster_falls_back_instead_of_generating_nothing(capsys, wired):
    # capsys BEFORE wired, deliberately: fixtures are set up in argument order
    # and capsys only captures from its own setup onward, so the other way
    # round the pipeline's whole run lands in the outer capture and
    # readouterr() returns an empty string that reads exactly like "the
    # warning was never printed".
    result, seen, _, _ = wired
    assert result.ok, result.message
    assert seen["reasoning_domains"]["deliberation"] is None, (
        "an empty roster must become domains=None - the old behaviour - not "
        "an empty list the generator would draw from")
    out = capsys.readouterr().out
    assert "CANNOT span" in out, (
        "the fallback weakens the specialist, so it has to say so: a corpus "
        "that quietly stopped spanning domains is a result nobody can explain "
        "afterwards")
