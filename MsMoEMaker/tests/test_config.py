"""Tests for ms_moe_maker/config.py — recipe -> PipelineConfig bridge."""

import pytest
from ms_moe_maker import config


class TestModelSizes:
    """MODEL_SIZES maps size strings to (base, abliterated) tuples."""

    def test_all_sizes_present(self):
        assert "0.5B" in config.MODEL_SIZES
        assert "1.5B" in config.MODEL_SIZES
        assert "3B" in config.MODEL_SIZES
        assert "7B" in config.MODEL_SIZES
        assert "14B" in config.MODEL_SIZES
        assert "32B" in config.MODEL_SIZES

    def test_each_has_two_elements(self):
        for size, pairs in config.MODEL_SIZES.items():
            assert len(pairs) == 2
            assert isinstance(pairs[0], str)
            assert isinstance(pairs[1], str)

    def test_0_5B_correct_bases(self):
        base, abliterated = config.MODEL_SIZES["0.5B"]
        assert "Qwen2.5-Coder-0.5B" in base
        assert "abliterated" in abliterated


class TestCodeLanguages:
    """CODE_LANGUAGES list of supported languages."""

    def test_has_four_languages(self):
        assert len(config.CODE_LANGUAGES) == 4

    def test_contains_python(self):
        assert "Python" in config.CODE_LANGUAGES

    def test_contains_csharp(self):
        assert "C#" in config.CODE_LANGUAGES

    def test_contains_powershell(self):
        assert "PowerShell" in config.CODE_LANGUAGES

    def test_contains_shell(self):
        assert "Shell" in config.CODE_LANGUAGES


class TestDisplayLang:
    """DISPLAY_LANG maps language keys to display names."""

    def test_python(self):
        assert config.DISPLAY_LANG["python"] == "Python"

    def test_csharp(self):
        assert config.DISPLAY_LANG["csharp"] == "C#"

    def test_powershell(self):
        assert config.DISPLAY_LANG["powershell"] == "PowerShell"

    def test_shell(self):
        assert config.DISPLAY_LANG["shell"] == "Bash"

    def test_missing_key_returns_none(self):
        assert config.DISPLAY_LANG.get("ruby") is None


class TestLanguageSources:
    """LANGUAGE_SOURCES maps languages to HuggingFace dataset specs."""

    def test_powershell_has_source(self):
        assert "PowerShell" in config.LANGUAGE_SOURCES

    def test_powershell_has_fields(self):
        src = config.LANGUAGE_SOURCES["PowerShell"]
        assert "repo" in src
        assert "split" in src
        assert "text_field" in src

    def test_powershell_repo_value(self):
        src = config.LANGUAGE_SOURCES["PowerShell"]
        assert src["repo"] == "SaeedRahmani/codeparrot_github_code_powershell"


class TestResolveRoots:
    """resolve_roots produces DATA_ROOT / OUTPUT_ROOT."""

    def test_dryrun_0_5b(self):
        roots = config.resolve_roots("0.5B", dryrun=True)
        assert roots["data"] == "msmoe_data"
        assert roots["output"] == "msmoe_dryrun_0.5B"

    def test_dryrun_7b(self):
        roots = config.resolve_roots("7B", dryrun=True)
        assert roots["data"] == "msmoe_data"
        assert roots["output"] == "msmoe_dryrun_7B"

    def test_production(self):
        roots = config.resolve_roots("1.5B", dryrun=False)
        assert roots["data"] == "msmoe_data"
        assert roots["output"] == "msmoe_run_1.5B"

    def test_no_run_root_is_named_after_another_project(self):
        """A user who pip-installs this and runs it in their own directory
        should not find a folder named after a repo they have never heard of.
        Same argument as data.code -> data.corpus in stages.py."""
        for size in ("0.5B", "32B"):
            for dry in (True, False):
                roots = config.resolve_roots(size, dryrun=dry)
                for v in roots.values():
                    assert "fraunkenstein" not in v.lower()
                    assert "qwen" not in v.lower()


class TestEnvBool:
    """_env_bool respects environment variables."""

    def test_true_values(self):
        import os
        old = os.environ.get("MSMOE_DRYRUN")
        try:
            os.environ["MSMOE_DRYRUN"] = "1"
            assert config._env_bool("MSMOE_DRYRUN", False) is True
            assert config._env_bool("MSMOE_DRYRUN", False) is True  # "true"
            os.environ["MSMOE_DRYRUN"] = "true"
            assert config._env_bool("MSMOE_DRYRUN", False) is True
        finally:
            if old is not None:
                os.environ["MSMOE_DRYRUN"] = old
            else:
                os.environ.pop("MSMOE_DRYRUN", None)

    def test_false_values(self):
        assert config._env_bool("MSMOE_NOPE", False) is False

    def test_default_when_unset(self):
        import os
        val = os.environ.pop("NONEXISTENT_KEY", None)
        try:
            assert config._env_bool("NONEXISTENT_KEY", False) is False
            assert config._env_bool("NONEXISTENT_KEY", True) is True
        finally:
            if val is not None:
                os.environ["NONEXISTENT_KEY"] = val


class TestPipelineConfigDefaults:
    """PipelineConfig frozen dataclass has sensible defaults."""

    def test_frozen(self):
        assert config.PipelineConfig.__dataclass_fields__["name"].metadata.get("frozen") or \
               config.PipelineConfig.__dataclass_params__.frozen

    def test_default_max_seq_length(self):
        pc = config.PipelineConfig(name="test", size="0.5B", base="test", base_safe="test")
        assert pc.max_seq_length == 2048

    def test_default_lora_r(self):
        pc = config.PipelineConfig(name="test", size="0.5B", base="test", base_safe="test")
        assert pc.lora_r == 64

    def test_default_target_steps(self):
        pc = config.PipelineConfig(name="test", size="0.5B", base="test", base_safe="test")
        assert pc.target_steps == 1200

    def test_default_router_mix_total(self):
        pc = config.PipelineConfig(name="test", size="0.5B", base="test", base_safe="test")
        assert pc.router_mix_total == 12_000

    def test_default_agents_mix_fraction(self):
        pc = config.PipelineConfig(name="test", size="0.5B", base="test", base_safe="test")
        assert pc.agent_mix_fraction == 0.15

    def test_default_expert_token_budget(self):
        pc = config.PipelineConfig(name="test", size="0.5B", base="test", base_safe="test")
        assert pc.expert_token_budget == 0  # computed, defaults to 0

    def test_default_dryrun_false(self):
        pc = config.PipelineConfig(name="test", size="0.5B", base="test", base_safe="test")
        assert pc.dryrun is False

    def test_default_force_false(self):
        pc = config.PipelineConfig(name="test", size="0.5B", base="test", base_safe="test")
        assert pc.force is False

    def test_default_llama_cpp_dir(self):
        pc = config.PipelineConfig(name="test", size="0.5B", base="test", base_safe="test")
        assert pc.llama_cpp_dir == "llama.cpp"

    def test_target_modules_default(self):
        pc = config.PipelineConfig(name="test", size="0.5B", base="test", base_safe="test")
        assert "gate_proj" in pc.target_modules
        assert "up_proj" in pc.target_modules
        assert "down_proj" in pc.target_modules

    def test_code_prompt_templates(self):
        pc = config.PipelineConfig(name="test", size="0.5B", base="test", base_safe="test")
        assert len(pc.code_prompt_templates) == 6
        assert "{lang}" in pc.code_prompt_templates[0]

    def test_code_prompt_unnamed(self):
        pc = config.PipelineConfig(name="test", size="0.5B", base="test", base_safe="test")
        assert len(pc.code_prompt_unnamed) == 3

    def test_expert_names_defaults_empty(self):
        pc = config.PipelineConfig(name="test", size="0.5B", base="test", base_safe="test")
        assert pc.expert_names == []


class TestBuildConfig:
    """build_config creates a PipelineConfig from a Recipe."""

    def test_invalid_size_auto_fills_from_tier(self):
        from ms_moe_maker.recipe import Recipe, Expert, Source
        from ms_moe_maker.config import MODEL_SIZES
        recipe = Recipe(
            name="test", base="Qwen/Qwen2.5-0.5B",
            experts=[Expert(
                name="python", source=Source(kind="local", path="/tmp")
            )],
            size="100TB",  # invalid — should auto-fill from tier default
        )
        pc = config.build_config(recipe)
        # Auto-filled from tier (spark → 3B)
        assert pc.size in MODEL_SIZES, f"size {pc.size} not in MODEL_SIZES"

    def test_valid_0_5b_resolves_base(self):
        from ms_moe_maker.recipe import Recipe, Expert, Source
        recipe = Recipe(
            name="test", base="Qwen/Qwen2.5-0.5B",
            experts=[Expert(
                name="python", source=Source(kind="code")
            )],
            size="0.5B",
        )
        pc = config.build_config(recipe)
        assert pc.size == "0.5B"
        assert "Qwen2.5-Coder-0.5B" in pc.base
        assert "abliterated" in pc.base
        assert "Qwen2.5-Coder-0.5B" in pc.base_safe

    def test_build_config_sets_data_and_output_roots(self):
        from ms_moe_maker.recipe import Recipe, Expert, Source
        import os
        old = os.environ.get("MSMOE_DRYRUN")
        try:
            os.environ["MSMOE_DRYRUN"] = "1"
            recipe = Recipe(
                name="test", base="Qwen/Qwen2.5-0.5B",
                experts=[Expert(
                    name="python", source=Source(kind="code")
                )],
                size="0.5B",
            )
            pc = config.build_config(recipe)
            roots = config.resolve_roots(pc.size, dryrun=True)
            assert pc.data_root == roots["data"]
            assert pc.output_root == roots["output"]
        finally:
            if old is not None:
                os.environ["MSMOE_DRYRUN"] = old
            else:
                os.environ.pop("MSMOE_DRYRUN", None)



class TestOneRunDirForEverybody:
    """Runner and build_config must resolve the same run directory.

    They did not. Runner called resolve_roots(recipe.size, ...) with the raw
    recipe size - "auto" whenever the recipe lets the tier pick - while
    build_config resolved "auto" to a concrete size first. Result: the manifest
    was written to msmoe_run_auto and every artifact to msmoe_run_7B, so a
    watcher reading the manifest and a resume looking for artifacts were
    pointed at two different empty-looking directories.
    """

    def _recipe(self, size="auto"):
        from ms_moe_maker.recipe import parse
        rec, _ = parse({
            "schema_version": 1, "name": "t", "size": size,
            "experts": [
                {"name": "a", "source": {"kind": "stack", "language": "Python"}},
                {"name": "b", "source": {"kind": "stack", "language": "C#"}},
            ],
        })
        return rec

    def test_auto_size_resolves_identically(self):
        rec = self._recipe("auto")
        roots = config.resolve_run_roots(rec)
        cfg = config.build_config(rec)
        assert roots["output"] == cfg.output_root
        assert roots["data"] == cfg.data_root

    def test_explicit_size_resolves_identically(self):
        rec = self._recipe("1.5B")
        roots = config.resolve_run_roots(rec)
        cfg = config.build_config(rec)
        assert roots["output"] == cfg.output_root

    def test_auto_never_leaks_into_a_path(self):
        """The tell that the two resolvers had diverged."""
        roots = config.resolve_run_roots(self._recipe("auto"))
        assert "auto" not in roots["output"]


class TestDryrunReachesTheBudgets:
    """`--dryrun` has to change what actually runs, not just a flag.

    It did not. build_config read MSMOE_DRYRUN and nothing else, so the CLI
    flag was inert: __main__ passed it to run_pipeline, which set
    `config.dryrun = True` AFTER build_config had resolved every budget from
    the environment — and no stage module reads config.dryrun at all.

    Asking for the cheap structural test therefore gave you the full corpus
    (100,000 samples instead of 10,000), the production minimum-samples floor,
    3x the router mix, AND it wrote into the production run directory, where a
    real run's artifacts live and where a resume would later find them.
    """

    def _recipe(self):
        from ms_moe_maker.recipe import parse
        rec, _ = parse({
            "schema_version": 1, "name": "t", "size": "0.5B",
            "experts": [
                {"name": "a", "source": {"kind": "hf", "repo": "o/d"}},
                {"name": "b", "source": {"kind": "hf", "repo": "o/e"}},
            ]})
        return rec

    def test_the_flag_shrinks_the_budgets(self, monkeypatch):
        monkeypatch.delenv("MSMOE_DRYRUN", raising=False)
        rec = self._recipe()
        full = config.build_config(rec, dryrun=False)
        dry = config.build_config(rec, dryrun=True)
        assert dry.num_code_samples < full.num_code_samples
        assert dry.min_samples_per_expert < full.min_samples_per_expert
        assert dry.router_mix_total < full.router_mix_total

    def test_the_flag_moves_the_run_directory(self, monkeypatch):
        """A dryrun must not write where a real run lives — otherwise a resume
        picks up structural-test artifacts as if they were the real thing."""
        monkeypatch.delenv("MSMOE_DRYRUN", raising=False)
        rec = self._recipe()
        assert "dryrun" in config.build_config(rec, dryrun=True).output_root
        assert "dryrun" not in config.build_config(rec, dryrun=False).output_root

    def test_runner_and_builder_agree_on_the_dryrun_directory(self, monkeypatch):
        monkeypatch.delenv("MSMOE_DRYRUN", raising=False)
        rec = self._recipe()
        for flag in (True, False):
            roots = config.resolve_run_roots(rec, dryrun=flag)
            cfg = config.build_config(rec, dryrun=flag)
            assert roots["output"] == cfg.output_root

    def test_the_env_var_still_works(self, monkeypatch):
        """MSMOE_DRYRUN is what the legacy subprocess path sets, and people
        have scripted it. None means 'ask the environment'."""
        monkeypatch.setenv("MSMOE_DRYRUN", "1")
        cfg = config.build_config(self._recipe())
        assert cfg.dryrun is True
        assert "dryrun" in cfg.output_root

    def test_an_explicit_false_beats_the_env_var(self, monkeypatch):
        monkeypatch.setenv("MSMOE_DRYRUN", "1")
        assert config.build_config(self._recipe(), dryrun=False).dryrun is False


class TestCorpusKnobs:
    """A recipe can ask for a REAL run that is merely small.

    Before this, corpus volume was hardcoded and the only lever was --dryrun -
    which also relabels the run as a structural test and moves it to a
    different output directory. So the thing a first end-to-end run wants
    most, "all the stages, real artifacts, small enough to watch it finish",
    was the one thing that could not be expressed.
    """

    def _rec(self, corpus=None):
        from ms_moe_maker.recipe import parse
        body = {"schema_version": 1, "name": "t", "size": "0.5B",
                "experts": [
                    {"name": "a", "source": {"kind": "hf", "repo": "o/d"}},
                    {"name": "b", "source": {"kind": "hf", "repo": "o/e"}}]}
        if corpus is not None:
            body["corpus"] = corpus
        rec, _ = parse(body)
        return rec

    def test_defaults_are_unchanged_when_unspecified(self, monkeypatch):
        """A recipe that says nothing must behave exactly as before."""
        monkeypatch.delenv("MSMOE_DRYRUN", raising=False)
        c = config.build_config(self._rec(), dryrun=False)
        assert c.min_samples_per_expert == 2_000
        assert c.num_code_samples == 100_000
        assert c.router_mix_total == 12_000

    def test_the_recipe_wins(self, monkeypatch):
        monkeypatch.delenv("MSMOE_DRYRUN", raising=False)
        c = config.build_config(self._rec(
            {"min_samples": 300, "max_samples": 3000,
             "router_mix_total": 800}), dryrun=False)
        assert (c.min_samples_per_expert, c.num_code_samples,
                c.router_mix_total) == (300, 3000, 800)

    def test_a_small_real_run_is_not_a_dryrun(self, monkeypatch):
        """The distinction that matters: small volume, production directory."""
        monkeypatch.delenv("MSMOE_DRYRUN", raising=False)
        c = config.build_config(self._rec({"max_samples": 3000}), dryrun=False)
        assert c.dryrun is False
        assert "dryrun" not in c.output_root

    def test_the_recipe_still_wins_under_dryrun(self, monkeypatch):
        monkeypatch.delenv("MSMOE_DRYRUN", raising=False)
        c = config.build_config(self._rec({"min_samples": 42}), dryrun=True)
        assert c.min_samples_per_expert == 42

    def test_minus_one_means_you_decide(self, monkeypatch):
        monkeypatch.delenv("MSMOE_DRYRUN", raising=False)
        explicit = config.build_config(
            self._rec({"min_samples": -1}), dryrun=True)
        implicit = config.build_config(self._rec(), dryrun=True)
        assert explicit.min_samples_per_expert == implicit.min_samples_per_expert

    # Measured on a DGX Spark: 2.56 s/optimiser step at 0.5B, seq 1024,
    # batch 4 x accum 2. Hardware-specific, and used ONLY to express the
    # shipped recipes' own promises about themselves in the unit those
    # promises are made in - minutes - rather than in a magic sample count.
    SECONDS_PER_STEP = 2.56
    FLOW_BUDGET_MIN = 45
    RUNG_BUDGET_MIN = 180

    def _shipped(self, name):
        import pathlib
        import pytest
        from ms_moe_maker.recipe import load, validate
        p = (pathlib.Path(config.__file__).parent.parent / name)
        if not p.is_file():
            pytest.skip(f"{name} not present in this checkout")
        rec, _ = load(str(p))
        errs, _ = validate(rec)
        assert errs == [], errs
        c = config.build_config(rec, dryrun=False)
        minutes = (c.target_steps * len(c.expert_names)
                   * self.SECONDS_PER_STEP / 60)
        return c, minutes

    def test_the_shipped_flow_recipe_validates_and_stays_a_shakedown(self):
        """THE PROMISE, ASSERTED IN THE UNIT IT IS MADE IN.

        This used to be `assert c.num_code_samples == 3000`, which is a magic
        number standing in for an intention. It caught the shipped recipe
        drifting into a two-hour run - correctly - and it would equally have
        snapped on any legitimate tuning, because equality on one knob cannot
        tell "someone made this bigger by accident" from "someone changed a
        different thing".

        The recipe's own header says 'small enough to watch it finish'. That
        is a claim about MINUTES, so assert minutes.
        """
        c, minutes = self._shipped("recipe.flow-0.5B.yaml")
        assert c.size == "0.5B"
        assert minutes < self.FLOW_BUDGET_MIN, (
            f"the flow recipe is a shakedown: {c.target_steps} steps x "
            f"{len(c.expert_names)} experts is ~{minutes:.0f} min of finetune, "
            f"over the {self.FLOW_BUDGET_MIN} min this file promises. Long "
            f"runs belong in recipe.rung-0.5B.yaml.")

    def test_the_shipped_rung_recipe_validates(self):
        """The rung is allowed to be slow - it is the measurement, not the
        smoke test - but it still has to be runnable as written, and it still
        has a ceiling so nobody ships a week-long example by accident."""
        c, minutes = self._shipped("recipe.rung-0.5B.yaml")
        assert c.size == "0.5B"
        assert len(c.expert_names) >= 3, (
            "the rung exists to make the routing claim provable, and 3 "
            "experts is the minimum width where p can clear 0.05")
        assert c.experts_per_tok == 2 and c.norm_topk_prob
        assert minutes < self.RUNG_BUDGET_MIN, f"~{minutes:.0f} min"

    def test_the_two_shipped_recipes_differ_only_in_budget(self):
        """If they drift apart on ARCHITECTURE, a result at one stops being
        evidence about the other - which is the entire premise of a ladder."""
        flow, _ = self._shipped("recipe.flow-0.5B.yaml")
        rung, _ = self._shipped("recipe.rung-0.5B.yaml")
        for field in ("experts_per_tok", "norm_topk_prob", "router_init",
                      "expert_names", "base", "max_seq_length",
                      "per_device_batch", "grad_accum"):
            assert getattr(flow, field) == getattr(rung, field), (
                f"{field} differs between the shakedown and the rung: "
                f"{getattr(flow, field)!r} vs {getattr(rung, field)!r}")
        assert rung.target_steps > flow.target_steps



def test_no_test_still_names_the_old_env_prefix():
    """The env-prefix cutover to MSMOE_ has to reach the tests too.

    A test that sets the OLD dryrun variable after the rename does not fail - it
    sets a variable nothing reads, and then asserts the DEFAULT behaviour while
    appearing to test the override. Green, and measuring nothing.
    """
    import pathlib
    here = pathlib.Path(__file__).resolve().parent
    # Built at runtime, so this file does not itself contain the literal it is
    # looking for - otherwise the guard flags its own docstring and the only
    # way to make it pass is to stop describing what it does.
    old = "FRAUNK" + "_"
    stale = []
    for f in here.glob("test_*.py"):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if old in line:
                stale.append(f"{f.name}:{i}")
    assert not stale, f"tests still using the retired prefix: {stale}"


def test_no_source_still_names_the_old_env_prefix():
    import pathlib
    pkg = pathlib.Path(config.__file__).parent
    old = "FRAUNK" + "_"
    stale = []
    for f in pkg.glob("*.py"):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if old in line:
                stale.append(f"{f.name}:{i}")
    assert not stale, f"source still using the retired prefix: {stale}"


# ── router knobs must reach the trainer ───────────────────────────────────
#
# The gate's learning rate was hardcoded in build_config while the class three
# doors down documented the opposite rule. The one experiment the first real
# 0.5B run called for - same stitch, more router training - could not be
# expressed in a recipe.

class TestRouterKnobs:

    def _data(self, **router):
        data = {
            "schema_version": 1, "name": "t", "size": "0.5B",
            "experts": [
                {"name": "a", "source": {"kind": "stack", "language": "Python"}},
                {"name": "b", "source": {"kind": "stack", "language": "C#"}},
            ],
        }
        if router:
            data["router"] = router
        return data

    def _cfg(self, tmp_path, **router):
        from ms_moe_maker.recipe import parse
        rec, _ = parse(self._data(**router))
        return config.build_config(rec, dryrun=True)

    def test_defaults_are_unchanged_when_the_recipe_is_silent(self, tmp_path):
        c = self._cfg(tmp_path)
        assert c.lr_router == 1e-4
        assert c.router_batch == 1
        assert c.router_accum == 8
        assert c.router_epochs == 1.0
        assert c.router_aux_loss_coef == 0.001

    def test_a_recipe_can_turn_every_router_knob(self, tmp_path):
        c = self._cfg(tmp_path, lr=1e-3, batch=4, accum=2, epochs=3,
                      aux_loss_coef=0.01)
        assert c.lr_router == 1e-3
        assert c.router_batch == 4
        assert c.router_accum == 2
        assert c.router_epochs == 3.0
        assert c.router_aux_loss_coef == 0.01

    def test_minus_one_still_means_you_decide(self, tmp_path):
        c = self._cfg(tmp_path, lr=-1, batch=-1, epochs=-1)
        assert c.lr_router == 1e-4
        assert c.router_batch == 1
        assert c.router_epochs == 1.0

    def test_router_block_is_a_known_top_level_key(self):
        """An unknown top-level key is warned and DROPPED. `eval` shipped in
        the README for months while not being in _KNOWN_TOP, so a user who
        wrote exactly what the docs said got their block silently ignored.
        Adding a dataclass is only half of adding a knob."""
        from ms_moe_maker.recipe import parse
        rec, warns = parse(self._data(lr=1e-3))
        assert not any("router" in w and "IGNORED" in w for w in warns), warns
        assert rec.router.lr == 1e-3


class TestTopOneRouterGradient:
    """top-1 + norm_topk_prob=true divides a weight by itself. Gate gets no
    gradient from the LM loss and can only drift toward uniform."""

    def _rec(self, top_k, norm, n_experts=2):
        from ms_moe_maker.recipe import parse
        langs = ["Python", "C#", "Go", "Rust"][:n_experts]
        rec, _ = parse({
            "schema_version": 1, "name": "t", "size": "0.5B",
            "experts": [{"name": f"e{i}",
                         "source": {"kind": "stack", "language": l}}
                        for i, l in enumerate(langs)],
            "moe": {"experts_per_tok": top_k, "norm_topk_prob": norm},
        })
        return rec

    def test_top1_with_normalisation_is_refused(self):
        from ms_moe_maker.recipe import validate
        errs, _ = validate(self._rec(1, True))
        assert any("severs the router from the loss" in e for e in errs), errs

    def test_top1_without_normalisation_is_fine(self):
        from ms_moe_maker.recipe import validate
        errs, _ = validate(self._rec(1, False))
        assert not any("norm_topk_prob" in e for e in errs), errs

    def test_top2_still_prefers_normalisation(self):
        from ms_moe_maker.recipe import validate
        errs, warns = validate(self._rec(2, False, n_experts=3))
        assert not any("severs" in e for e in errs), errs
        assert any("0.40x at init" in w for w in warns), warns

    def test_the_degenerate_hint_names_both_fields(self):
        """The hint that sends you to top-1 must also send you to
        norm_topk_prob=false, or it walks you into the other trap."""
        from ms_moe_maker.recipe import validate
        _, warns = validate(self._rec(2, True, n_experts=2))
        hint = [w for w in warns if "experts_per_tok=2" in w]
        assert hint, warns
        assert "norm_topk_prob=false" in hint[0], hint[0]


class TestLoraKnobs:
    """Rank was reachable only from an env var, and the line that looked like
    it read the recipe assigned `target_steps` to `lora_r` before the tier
    overwrote it. Dead in both paths, and misleading to read: raising
    target_steps looks like it should raise the rank, and never did."""

    def _rec(self, **budget):
        from ms_moe_maker.recipe import parse
        data = {
            "schema_version": 1, "name": "t", "size": "0.5B",
            "experts": [{"name": "a",
                         "source": {"kind": "stack", "language": "Python"}}],
            "runtime": {"hardware_tier": "spark"},
        }
        if budget:
            data["budget"] = budget
        rec, _ = parse(data)
        return rec

    def test_the_tier_default_still_applies_when_the_recipe_is_silent(self):
        assert config.build_config(self._rec(), dryrun=True).lora_r == 128

    def test_a_recipe_can_set_the_rank(self):
        c = config.build_config(self._rec(lora_r=192), dryrun=True)
        assert c.lora_r == 192

    def test_target_steps_does_not_move_the_rank(self):
        """The exact confusion the old line created."""
        c = config.build_config(self._rec(target_steps=600), dryrun=True)
        assert c.lora_r == 128, (
            "target_steps is a schedule length, not an adapter rank")
        assert c.target_steps == 600

    def test_alpha_and_dropout_are_reachable(self):
        c = config.build_config(self._rec(lora_alpha=64, lora_dropout=0.05),
                                dryrun=True)
        assert c.lora_alpha == 64
        assert c.lora_dropout == 0.05

    def test_alpha_defaults_are_unchanged(self):
        c = config.build_config(self._rec(), dryrun=True)
        assert c.lora_alpha == 32
        assert c.lora_dropout == 0.0

    def test_the_rank_is_still_capped(self):
        assert config.build_config(self._rec(lora_r=9999),
                                   dryrun=True).lora_r == 256
