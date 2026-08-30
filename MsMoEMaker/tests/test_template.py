"""Tests for the recipe template system."""
import pytest

from ms_moe_maker.config import templates as template


class TestGetTemplate:
    """get_template(name) returns the template dict."""

    def test_code_template(self):
        tpl = template.get_template("code")
        assert tpl is not None
        assert tpl["preferred_size"] == "3B"
        assert tpl["base_hint"] == "Qwen/Qwen2.5-Coder"
        assert tpl["default_tier"] == "spark"
        assert len(tpl["default_experts"]) == 2

    def test_dnd_template(self):
        tpl = template.get_template("dnd")
        assert tpl is not None
        assert tpl["preferred_size"] == "0.5B"
        assert tpl["default_tier"] == "nano"
        assert len(tpl["default_experts"]) == 3

    def test_math_template(self):
        tpl = template.get_template("math")
        assert tpl is not None
        assert tpl["preferred_size"] == "1.5B"
        assert len(tpl["default_experts"]) == 4

    def test_culinary_template(self):
        tpl = template.get_template("culinary")
        assert tpl is not None
        assert tpl["preferred_size"] == "1.5B"
        assert len(tpl["default_experts"]) == 3

    def test_unknown_returns_none(self):
        assert template.get_template("fantasy") is None


class TestDescribeTemplates:
    """describe_templates() returns summary dict."""

    def test_has_all_templates(self):
        desc = template.describe_templates()
        assert set(desc.keys()) == {"code", "dnd", "math", "culinary"}

    def test_dnd_tier(self):
        desc = template.describe_templates()
        assert desc["dnd"]["default_tier"] == "nano"

    def test_math_expert_count(self):
        desc = template.describe_templates()
        assert desc["math"]["expert_count"] == 4

    def test_culinary_expert_count(self):
        desc = template.describe_templates()
        assert desc["culinary"]["expert_count"] == 3


class TestApplyTemplate:
    """apply_template(recipe, name) merges template into recipe dict."""

    def test_fills_missing_fields(self):
        """A template fills the RECIPE's fields, not its own metadata.

        This test used to assert merged["preferred_size"] and
        merged["default_tier"] - i.e. that the template's own notes survived
        into the recipe. They did, and parse() then reported every one of them
        as an unknown top-level key, so `init --template dnd` produced a file
        that validated with ten warnings the user had not caused and could not
        act on. Now the metadata is TRANSLATED into where it belongs and then
        dropped.
        """
        recipe = {"name": "my-moe", "size": "3B"}
        merged = template.apply_template(recipe, "code")

        # translated into real recipe fields
        assert merged["size"] == "3B"                       # recipe's own wins
        assert merged["runtime"]["hardware_tier"] == "spark"
        assert "_base_hint" in merged                        # internal, kept
        assert "experts" in merged                           # default_experts

        # and the template's own vocabulary is gone
        for internal in ("preferred_size", "default_tier", "default_experts",
                         "default_moe", "default_runtime", "default_budget",
                         "base_hint"):
            assert internal not in merged, (
                f"{internal} is template metadata, not recipe content - "
                f"leaving it makes parse() warn about a key the user never "
                f"wrote")

    def test_preferred_size_becomes_size_when_unset(self):
        merged = template.apply_template({}, "dnd")
        assert merged["size"] == template.TEMPLATES["dnd"]["preferred_size"]

    def test_every_template_produces_a_valid_recipe(self):
        """A shipped template that cannot validate is a broken on-ramp, and it
        breaks at the worst moment: the user's first command. `culinary`
        shipped a local source with an empty path and failed exactly there."""
        from ms_moe_maker.config.recipe import parse, validate
        for name in template.TEMPLATES:
            rec, _ = parse({"schema_version": 1, "template": name})
            errs, _ = validate(rec)
            assert errs == [], f"template {name!r} does not validate: {errs}"

    def test_a_recipe_budget_merges_into_the_template_block(self):
        """Per-key merge, not wholesale clobber. `budget: {target_steps: 300}`
        used to drop the code template's max_seq_length 4096 and fall to the
        dataclass 2048, silently halving tokens/expert."""
        merged = template.apply_template(
            {"template": "code", "budget": {"target_steps": 300}}, "code")
        assert merged["budget"]["target_steps"] == 300
        assert merged["budget"]["max_seq_length"] == 4096

    def test_templates_keep_the_dead_expert_check_armed(self):
        """dead_threshold below 1.0 can never flag anything - enrichment
        bottoms out at 1.0 (uniform routing). The README default is 1.2."""
        for name in template.TEMPLATES:
            tpl = template.TEMPLATES[name]
            assert tpl["default_eval"]["dead_threshold"] == 1.2, name

    def test_recipe_wins_over_template(self):
        recipe = {
            "size": "0.5B",
            "preferred_size": "0.5B",
            "experts": [{"name": "custom", "source": {"kind": "stack", "language": "Python"}}],
        }
        merged = template.apply_template(recipe, "code")
        assert merged["size"] == "0.5B"
        assert merged["experts"] == recipe["experts"]

    def test_unknown_template_raises(self):
        with pytest.raises(ValueError, match="unknown template"):
            template.apply_template({}, "fantasy")

    def test_nested_budget_filled(self):
        recipe = {"name": "test"}
        merged = template.apply_template(recipe, "dnd")
        assert "budget" in merged
        assert merged["budget"]["target_steps"] == 500

    def test_nested_moe_filled(self):
        recipe = {"name": "test"}
        merged = template.apply_template(recipe, "code")
        assert "moe" in merged
        assert merged["moe"]["experts_per_tok"] == 2


class TestTemplateSources:
    """Source configs within templates."""

    def test_code_has_stack_sources(self):
        tpl = template.get_template("code")
        kinds = [e["source"]["kind"] for e in tpl["default_experts"]]
        assert "stack" in kinds

    def test_dnd_has_hf_sources(self):
        tpl = template.get_template("dnd")
        kinds = [e["source"]["kind"] for e in tpl["default_experts"]]
        assert all(k == "hf" for k in kinds)

    def test_dnd_expert_names(self):
        tpl = template.get_template("dnd")
        names = [e["name"] for e in tpl["default_experts"]]
        assert "monster_manual" in names
        assert "dm_guide" in names
        assert "players_handbook" in names

    def test_math_expert_names(self):
        tpl = template.get_template("math")
        names = [e["name"] for e in tpl["default_experts"]]
        assert "arithmetic" in names
        assert "algebra" in names
        assert "geometry" in names
        assert "word_problems" in names
