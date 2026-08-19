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
        old = os.environ.get("FRAUNK_DRYRUN")
        try:
            os.environ["FRAUNK_DRYRUN"] = "1"
            assert config._env_bool("FRAUNK_DRYRUN", False) is True
            assert config._env_bool("FRAUNK_DRYRUN", False) is True  # "true"
            os.environ["FRAUNK_DRYRUN"] = "true"
            assert config._env_bool("FRAUNK_DRYRUN", False) is True
        finally:
            if old is not None:
                os.environ["FRAUNK_DRYRUN"] = old
            else:
                os.environ.pop("FRAUNK_DRYRUN", None)

    def test_false_values(self):
        assert config._env_bool("FRAUNK_NOPE", False) is False

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
        old = os.environ.get("FRAUNK_DRYRUN")
        try:
            os.environ["FRAUNK_DRYRUN"] = "1"
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
                os.environ["FRAUNK_DRYRUN"] = old
            else:
                os.environ.pop("FRAUNK_DRYRUN", None)



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

    It did not. build_config read FRAUNK_DRYRUN and nothing else, so the CLI
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
        monkeypatch.delenv("FRAUNK_DRYRUN", raising=False)
        rec = self._recipe()
        full = config.build_config(rec, dryrun=False)
        dry = config.build_config(rec, dryrun=True)
        assert dry.num_code_samples < full.num_code_samples
        assert dry.min_samples_per_expert < full.min_samples_per_expert
        assert dry.router_mix_total < full.router_mix_total

    def test_the_flag_moves_the_run_directory(self, monkeypatch):
        """A dryrun must not write where a real run lives — otherwise a resume
        picks up structural-test artifacts as if they were the real thing."""
        monkeypatch.delenv("FRAUNK_DRYRUN", raising=False)
        rec = self._recipe()
        assert "dryrun" in config.build_config(rec, dryrun=True).output_root
        assert "dryrun" not in config.build_config(rec, dryrun=False).output_root

    def test_runner_and_builder_agree_on_the_dryrun_directory(self, monkeypatch):
        monkeypatch.delenv("FRAUNK_DRYRUN", raising=False)
        rec = self._recipe()
        for flag in (True, False):
            roots = config.resolve_run_roots(rec, dryrun=flag)
            cfg = config.build_config(rec, dryrun=flag)
            assert roots["output"] == cfg.output_root

    def test_the_env_var_still_works(self, monkeypatch):
        """FRAUNK_DRYRUN is what the legacy subprocess path sets, and people
        have scripted it. None means 'ask the environment'."""
        monkeypatch.setenv("FRAUNK_DRYRUN", "1")
        cfg = config.build_config(self._recipe())
        assert cfg.dryrun is True
        assert "dryrun" in cfg.output_root

    def test_an_explicit_false_beats_the_env_var(self, monkeypatch):
        monkeypatch.setenv("FRAUNK_DRYRUN", "1")
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
        monkeypatch.delenv("FRAUNK_DRYRUN", raising=False)
        c = config.build_config(self._rec(), dryrun=False)
        assert c.min_samples_per_expert == 2_000
        assert c.num_code_samples == 100_000
        assert c.router_mix_total == 12_000

    def test_the_recipe_wins(self, monkeypatch):
        monkeypatch.delenv("FRAUNK_DRYRUN", raising=False)
        c = config.build_config(self._rec(
            {"min_samples": 300, "max_samples": 3000,
             "router_mix_total": 800}), dryrun=False)
        assert (c.min_samples_per_expert, c.num_code_samples,
                c.router_mix_total) == (300, 3000, 800)

    def test_a_small_real_run_is_not_a_dryrun(self, monkeypatch):
        """The distinction that matters: small volume, production directory."""
        monkeypatch.delenv("FRAUNK_DRYRUN", raising=False)
        c = config.build_config(self._rec({"max_samples": 3000}), dryrun=False)
        assert c.dryrun is False
        assert "dryrun" not in c.output_root

    def test_the_recipe_still_wins_under_dryrun(self, monkeypatch):
        monkeypatch.delenv("FRAUNK_DRYRUN", raising=False)
        c = config.build_config(self._rec({"min_samples": 42}), dryrun=True)
        assert c.min_samples_per_expert == 42

    def test_minus_one_means_you_decide(self, monkeypatch):
        monkeypatch.delenv("FRAUNK_DRYRUN", raising=False)
        explicit = config.build_config(
            self._rec({"min_samples": -1}), dryrun=True)
        implicit = config.build_config(self._rec(), dryrun=True)
        assert explicit.min_samples_per_expert == implicit.min_samples_per_expert

    def test_the_shipped_flow_recipe_validates(self):
        """The 0.5B end-to-end recipe has to be runnable as written."""
        import pathlib
        from ms_moe_maker.recipe import load, validate
        p = (pathlib.Path(config.__file__).parent.parent
             / "recipe.flow-0.5B.yaml")
        if not p.is_file():
            import pytest
            pytest.skip("flow recipe not present in this checkout")
        rec, _ = load(str(p))
        errs, _ = validate(rec)
        assert errs == [], errs
        c = config.build_config(rec, dryrun=False)
        assert c.size == "0.5B"
        assert c.num_code_samples == 3000, "the flow recipe must stay small"
