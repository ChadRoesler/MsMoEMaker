"""The defaults layer: the box configures the floor, the recipe always wins.

The case this exists for is someone setting a machine up for someone else -
picking sensible values for THAT hardware once, so their recipes stay six
lines. The risk it introduces is that a recipe stops fully describing its own
build, which is why provenance is tested as hard as the merge is.
"""
import json
import os

import pytest

from ms_moe_maker import defaults as D
from ms_moe_maker import recipe as R


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return str(p)


class TestResolve:
    def test_the_floor_alone_is_enough(self, monkeypatch, tmp_path):
        """No files anywhere: the tool still runs. The floor is a panic
        minimum, not a convenience."""
        monkeypatch.setenv(D.USER_ENV, str(tmp_path / "nope.yaml"))
        monkeypatch.setattr(D, "_packaged_path", lambda: str(tmp_path / "gone.yaml"))
        d, prov, warns = D.resolve()
        assert d["tools_expert"]["name"]
        assert d["tools_expert"]["teacher"]
        assert warns == []

    def test_the_packaged_file_ships_and_parses(self):
        assert os.path.isfile(D._packaged_path()), (
            "defaults.yaml is the real table; a wheel without it behaves "
            "differently from a checkout")
        d, _, warns = D.resolve(include_user=False)
        assert warns == []
        assert d["tools_expert"]["name"]

    def test_the_packaged_file_does_not_contradict_the_floor(self):
        """Two copies of one fact is the bug this codebase keeps finding. The
        floor is allowed to be SMALLER than the shipped table; it is not
        allowed to disagree with it."""
        d, _, _ = D.resolve(include_user=False)
        for block, values in D.FLOOR.items():
            for k, v in values.items():
                assert d[block][k] == v, (
                    f"{block}.{k}: floor says {v!r}, packaged defaults.yaml "
                    f"says {d[block][k]!r}")

    def test_a_later_layer_wins_and_says_so(self, tmp_path):
        box = _write(tmp_path, "box.yaml",
                     "budget:\n  target_steps: 400\n")
        d, prov, _ = D.resolve(box, include_user=False)
        assert d["budget"]["target_steps"] == 400
        assert prov["budget.target_steps"] == box

    def test_minus_one_falls_through(self, tmp_path):
        """`-1` has meant 'you decide' since the first recipe. The layer below
        is now who decides, so -1 must not overwrite it with -1."""
        low = _write(tmp_path, "low.yaml", "budget:\n  target_steps: 400\n")
        d, _, _ = D.resolve(low, include_user=False)
        merged = D.apply_to({"budget": {"target_steps": -1}}, d)
        assert merged["budget"]["target_steps"] == 400

    def test_a_defaults_file_cannot_inject_experts(self, tmp_path):
        bad = _write(tmp_path, "bad.yaml",
                     "experts:\n  - name: sneaky\nname: nope\n")
        d, _, warns = D.resolve(bad, include_user=False)
        assert "experts" not in d and "name" not in d
        assert any("experts" in w for w in warns), warns

    def test_an_unreadable_explicit_file_is_loud(self, tmp_path):
        d, _, warns = D.resolve(str(tmp_path / "missing.yaml"),
                                include_user=False)
        assert any("could not be read" in w for w in warns), warns

    def test_the_user_layer_can_be_excluded(self, tmp_path, monkeypatch):
        """A unit test whose result depends on whoever's laptop it runs on is
        not a unit test."""
        mine = _write(tmp_path, "mine.yaml", "budget:\n  target_steps: 999\n")
        monkeypatch.setenv(D.USER_ENV, mine)
        with_user, _, _ = D.resolve(include_user=True)
        without, _, _ = D.resolve(include_user=False)
        assert with_user.get("budget", {}).get("target_steps") == 999
        assert "budget" not in without


class TestContentOnlyBlocks:
    """Saying what a thing IS is not asking for one."""

    def test_defaults_alone_do_not_add_a_tools_expert(self, tmp_path):
        """THE BUG THIS PINS. With a raw merge, `tools_expert:` sitting in a
        defaults file made every recipe on the box grow an agentcore expert -
        an extra specialist, its teacher, and its GPU hours - injected by a
        file the recipe never mentions."""
        rec_path = _write(tmp_path, "r.json", json.dumps({
            "schema_version": 1, "name": "t", "size": "0.5B",
            "experts": [{"name": "a", "source": {"kind": "hf", "repo": "o/d"}},
                        {"name": "b", "source": {"kind": "hf", "repo": "o/e"}}],
        }))
        rec, _ = R.load(rec_path, include_user_defaults=False)
        assert [e.name for e in rec.experts] == ["a", "b"]
        assert getattr(rec, "tools_expert_name", "") == ""

    def test_asking_for_one_fills_it_from_the_box(self, tmp_path):
        box = _write(tmp_path, "box.yaml",
                     "tools_expert:\n  name: toolsy\n"
                     "  teacher: Qwen/Qwen2.5-7B-Instruct\n")
        rec_path = tmp_path / "r.json"
        rec_path.write_text(json.dumps({
            "schema_version": 1, "name": "t", "size": "0.5B",
            "tools_expert": True,
            "experts": [{"name": "a", "source": {"kind": "hf", "repo": "o/d"}},
                        {"name": "b", "source": {"kind": "hf", "repo": "o/e"}}],
        }), encoding="utf-8")
        rec, _ = R.load(str(rec_path), defaults_path=box,
                        include_user_defaults=False)
        names = [e.name for e in rec.experts]
        assert "toolsy" in names, names
        assert rec.tools_expert_name == "toolsy"
        tools = next(e for e in rec.experts if e.name == "toolsy")
        assert tools.source.kind == "synth"
        assert tools.source.teacher == "Qwen/Qwen2.5-7B-Instruct"

    def test_the_recipe_still_overrides_the_box(self, tmp_path):
        box = _write(tmp_path, "box.yaml",
                     "tools_expert:\n  name: toolsy\n  teacher: box/teacher\n")
        rec_path = tmp_path / "r.json"
        rec_path.write_text(json.dumps({
            "schema_version": 1, "name": "t", "size": "0.5B",
            "tools_expert": {"teacher": "recipe/teacher"},
            "experts": [{"name": "a", "source": {"kind": "hf", "repo": "o/d"}},
                        {"name": "b", "source": {"kind": "hf", "repo": "o/e"}}],
        }), encoding="utf-8")
        rec, _ = R.load(str(rec_path), defaults_path=box,
                        include_user_defaults=False)
        tools = next(e for e in rec.experts if e.name == "toolsy")
        assert tools.source.teacher == "recipe/teacher"


class TestProvenance:
    def test_every_applied_default_names_its_file(self, tmp_path):
        box = _write(tmp_path, "box.yaml",
                     "budget:\n  target_steps: 400\ncorpus:\n  max_samples: 9000\n")
        rec_path = tmp_path / "r.json"
        rec_path.write_text(json.dumps({
            "schema_version": 1, "name": "t", "size": "0.5B",
            "experts": [{"name": "a", "source": {"kind": "hf", "repo": "o/d"}},
                        {"name": "b", "source": {"kind": "hf", "repo": "o/e"}}],
        }), encoding="utf-8")
        rec, _ = R.load(str(rec_path), defaults_path=box,
                        include_user_defaults=False)
        prov = rec.defaults_provenance
        assert prov["budget.target_steps"] == box
        assert prov["corpus.max_samples"] == box

    def test_an_unrequested_content_block_is_not_reported_as_applied(
            self, tmp_path):
        """Reporting more than we know, pointed at our own output."""
        box = _write(tmp_path, "box.yaml", "tools_expert:\n  teacher: x/y\n")
        rec_path = tmp_path / "r.json"
        rec_path.write_text(json.dumps({
            "schema_version": 1, "name": "t", "size": "0.5B",
            "experts": [{"name": "a", "source": {"kind": "hf", "repo": "o/d"}},
                        {"name": "b", "source": {"kind": "hf", "repo": "o/e"}}],
        }), encoding="utf-8")
        rec, _ = R.load(str(rec_path), defaults_path=box,
                        include_user_defaults=False)
        assert not any(k.startswith("tools_expert")
                       for k in rec.defaults_provenance)


class TestParseStaysPure:
    def test_parse_ignores_the_box_entirely(self, tmp_path, monkeypatch):
        """parse() takes a dict, so it must be reproducible anywhere. Only
        load() - which touches a path - is allowed to touch a machine."""
        mine = _write(tmp_path, "mine.yaml",
                      "tools_expert:\n  name: fromlaptop\n")
        monkeypatch.setenv(D.USER_ENV, mine)
        rec, _ = R.parse({
            "schema_version": 1, "name": "t", "size": "0.5B",
            "tools_expert": True,
            "experts": [{"name": "a", "source": {"kind": "hf", "repo": "o/d"}}],
        })
        assert "fromlaptop" not in [e.name for e in rec.experts]


class TestBoxTiers:
    """`tiers:` and `models:` are statements about a MACHINE.

    A recipe may name a tier. It must never redefine one, or the same recipe
    describes different hardware depending on who ran it.
    """

    def _recipe(self, tmp_path, **extra):
        body = {"schema_version": 1, "name": "t",
                "experts": [{"name": "a", "source": {"kind": "hf", "repo": "o/d"}},
                            {"name": "b", "source": {"kind": "hf", "repo": "o/e"}}]}
        body.update(extra)
        p = tmp_path / "r.json"
        p.write_text(json.dumps(body), encoding="utf-8")
        return str(p)

    def test_a_box_can_redefine_a_tier(self, tmp_path):
        from ms_moe_maker import config as C
        box = _write(tmp_path, "box.yaml",
                     "tiers:\n  spark:\n    default_size: 14B\n"
                     "    default_lora_r: 96\n")
        rec, _ = R.load(self._recipe(tmp_path, runtime={"hardware_tier": "spark"}),
                        defaults_path=box, include_user_defaults=False)
        spec = C.tier_table(rec)["spark"]
        assert (spec.default_size, spec.default_lora_r) == ("14B", 96)
        # and the untouched fields survived
        assert spec.default_quant == "Q8_0"
        cfg = C.build_config(rec, dryrun=False)
        assert cfg.size == "14B"
        assert cfg.lora_r == 96

    def test_a_box_can_add_a_tier_the_tool_never_heard_of(self, tmp_path):
        from ms_moe_maker import config as C
        box = _write(tmp_path, "box.yaml",
                     "runtime:\n  hardware_tier: orin_agx\n"
                     "tiers:\n  orin_agx:\n    like: spark\n"
                     "    max_vram_gb: 64\n    default_size: 7B\n"
                     "    default_quant: Q5_K_M\n")
        rec, warns = R.load(self._recipe(tmp_path), defaults_path=box,
                            include_user_defaults=False)
        cfg = C.build_config(rec, dryrun=False)
        assert cfg.tier == "orin_agx"
        assert cfg.size == "7B"
        assert cfg.lora_r == 128, "like: spark should carry the rank over"
        assert not any("not a tier on this box" in w for w in warns), warns

    def test_a_new_tier_missing_fields_is_refused_out_loud(self, tmp_path):
        """Refusing is fine. Refusing silently is not - merge_tiers does not
        raise, so the warning is the only thing standing between a typo and a
        build that quietly used the middle tier."""
        box = _write(tmp_path, "box.yaml",
                     "tiers:\n  halfbaked:\n    max_vram_gb: 12\n")
        _, warns = R.load(self._recipe(tmp_path), defaults_path=box,
                          include_user_defaults=False)
        assert any("halfbaked" in w and "missing" in w for w in warns), warns

    def test_naming_a_tier_the_box_does_not_have_is_said_out_loud(self, tmp_path):
        box = _write(tmp_path, "box.yaml", "budget:\n  target_steps: 400\n")
        _, warns = R.load(
            self._recipe(tmp_path, runtime={"hardware_tier": "orin_agx"}),
            defaults_path=box, include_user_defaults=False)
        assert any("not a tier on this box" in w for w in warns), warns

    def test_an_unknown_tier_field_is_ignored_out_loud(self, tmp_path):
        box = _write(tmp_path, "box.yaml",
                     "tiers:\n  spark:\n    default_sizee: 14B\n")
        _, warns = R.load(self._recipe(tmp_path), defaults_path=box,
                          include_user_defaults=False)
        assert any("default_sizee" in w for w in warns), warns

    def test_a_box_can_point_a_size_at_its_own_checkpoint(self, tmp_path):
        from ms_moe_maker import config as C
        box = _write(tmp_path, "box.yaml",
                     'models:\n  "0.5B":\n'
                     "    abliterated: /mnt/models/local-0.5B\n")
        rec, _ = R.load(self._recipe(tmp_path, size="0.5B"), defaults_path=box,
                        include_user_defaults=False)
        assert C.model_sizes(rec)["0.5B"][1] == "/mnt/models/local-0.5B"
        assert C.build_config(rec, dryrun=False).base == "/mnt/models/local-0.5B"

    def test_a_bare_string_is_the_kind_reading(self, tmp_path):
        from ms_moe_maker import config as C
        box = _write(tmp_path, "box.yaml", 'models:\n  "0.5B": /mnt/m/tiny\n')
        rec, _ = R.load(self._recipe(tmp_path, size="0.5B"), defaults_path=box,
                        include_user_defaults=False)
        assert C.model_sizes(rec)["0.5B"] == ("/mnt/m/tiny", "/mnt/m/tiny")

    def test_the_recipe_cannot_redefine_hardware(self, tmp_path):
        """`tiers:` in a RECIPE is not a recipe key and must be ignored, loudly.
        Otherwise a recipe stops being portable in the one way that matters."""
        rec, warns = R.load(
            self._recipe(tmp_path, tiers={"spark": {"default_size": "0.5B"}}),
            include_user_defaults=False)
        from ms_moe_maker import config as C
        assert C.tier_table(rec)["spark"].default_size == "32B"
        assert any("tiers" in w and "IGNORED" in w for w in warns), warns

    def test_parse_resolves_against_the_floor_alone(self, tmp_path, monkeypatch):
        from ms_moe_maker import config as C
        mine = _write(tmp_path, "mine.yaml",
                      "tiers:\n  spark:\n    default_size: 0.5B\n")
        monkeypatch.setenv(D.USER_ENV, mine)
        rec, _ = R.parse({"schema_version": 1, "name": "t",
                          "experts": [{"name": "a",
                                       "source": {"kind": "hf", "repo": "o/d"}}]})
        assert C.tier_table(rec)["spark"].default_size == "32B"


class TestTheOnRamp:
    """`init --defaults-template` — the on-ramp for the BOX, not the build."""

    def _run(self, argv):
        from ms_moe_maker.__main__ import main
        return main(argv)

    def test_the_template_it_writes_is_a_valid_defaults_file(self, tmp_path):
        """THE BUG THIS PINS, and it is the same shape as the one the
        init/validate round-trip caught: the on-ramp tripping over its own
        first step. The template is entirely comments, yaml.safe_load returns
        None for that, and `read_yaml` called it unreadable — so the very next
        use of a freshly generated file warned that it could not be read.
        Empty means empty.
        """
        dest = tmp_path / "defaults.yaml"
        assert self._run(["init", "--defaults-template",
                          "--output", str(dest)]) == 0
        assert dest.is_file()
        _, _, warns = D.resolve(str(dest), include_user=False)
        assert warns == [], warns

    def test_every_commented_key_in_the_template_is_a_real_key(self, tmp_path):
        """A starter file that teaches a key the parser ignores is worse than
        no starter file."""
        import re
        from ms_moe_maker.__main__ import _defaults_template_body
        body = _defaults_template_body()
        tops = set(re.findall(r"^# ([a-z_]+):\s*$", body, re.M))
        assert tops, "the template stopped containing any blocks"
        assert tops <= D.ALLOWED, sorted(tops - D.ALLOWED)

    def test_it_refuses_to_clobber_a_machines_configuration(self, tmp_path):
        dest = tmp_path / "defaults.yaml"
        assert self._run(["init", "--defaults-template",
                          "--output", str(dest)]) == 0
        dest.write_text("budget: {target_steps: 42}\n", encoding="utf-8")
        assert self._run(["init", "--defaults-template",
                          "--output", str(dest)]) == 1
        assert "42" in dest.read_text(encoding="utf-8"), "it overwrote anyway"
        assert self._run(["init", "--defaults-template", "--force",
                          "--output", str(dest)]) == 0

    def test_an_empty_file_is_empty_not_broken(self, tmp_path):
        blank = tmp_path / "blank.yaml"
        blank.write_text("# nothing here yet\n", encoding="utf-8")
        resolved, _, warns = D.resolve(str(blank), include_user=False)
        assert warns == []
        assert resolved["tools_expert"]["name"], "the layers below survived"

    def test_a_malformed_file_is_still_loud(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("- this is a list, not a mapping\n", encoding="utf-8")
        _, _, warns = D.resolve(str(bad), include_user=False)
        assert any("could not be read" in w for w in warns), warns
