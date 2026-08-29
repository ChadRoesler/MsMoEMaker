"""Tests for stitching — the stage that had none, and needed them most.

stitch_moe could not complete under ANY configuration and nothing caught it,
because no test in the suite touched this module. Three independent failures:

  1. `config_dict.pop("quantization_config")` with no default raised KeyError
     on a NON-quantised specialist, which is the only kind that can be
     stitched. The fail-fast guard failed on the happy path.
  2. `import moe_stitch` was bound LOCAL to stitch_moe while _stream_stitch
     looked the name up as a global -> NameError, which `except ImportError`
     does not catch.
  3. The in-process fallback looked up `mlp.experts.N.*` keys inside a DENSE
     specialist checkpoint, where they cannot exist, so it copied no expert
     weights at all.

Each of these gets a test here.
"""
import json
import types

import pytest

from ms_moe_maker.moe import stitch
from ms_moe_maker.run import stages as st


class TestQuantisationGuard:
    """The guard that crashed on the configuration it exists to allow."""

    def test_absent_key_is_not_an_error(self):
        cfg = {"model_type": "qwen2", "num_hidden_layers": 24}
        # The shape the code uses now. Must not raise.
        assert cfg.pop("quantization_config", None) is None

    def test_the_old_shape_would_have_raised(self):
        """Documents the bug so nobody reintroduces the no-default pop."""
        cfg = {"model_type": "qwen2"}
        with pytest.raises(KeyError):
            cfg.pop("quantization_config")

    def test_source_has_no_bare_pop(self):
        import inspect
        src = inspect.getsource(stitch)
        assert 'pop("quantization_config")' not in src, (
            "a bare pop here raises KeyError for every non-quantised "
            "specialist, i.e. every specialist that can actually be stitched")


class TestOnePathOnly:
    """The dead fallback is gone and must stay gone."""

    def test_no_inproc_fallback_exists(self):
        assert not hasattr(stitch, "_inproc_stitch"), (
            "the in-process fallback could not copy expert weights - it looked "
            "for mlp.experts.N.* keys in a dense specialist checkpoint. It "
            "would emit an MoE whose every expert is a copy of the anchor, "
            "which a config-only verify would wave straight through")

    def test_vendored_stitcher_ships_with_the_package(self):
        """The whole point of vendoring: no private repo beside the install.

        Checked by parsing the file rather than importing it, so the test
        holds on a machine with no torch - which is precisely the machine
        where the missing-module bug used to hide.
        """
        import ast
        import pathlib
        f = pathlib.Path(stitch.__file__).with_name("_moe_stitch.py")
        assert f.is_file(), "_moe_stitch.py is not shipped with the package"
        tree = ast.parse(f.read_text(encoding="utf-8"))
        defined = {n.name for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for fn in ("stream_stitch", "plan_from_meta", "safetensors_map",
                   "ShardWriter"):
            assert fn in defined, f"vendored module is missing {fn}"

    def test_stitch_does_not_import_a_bare_moe_stitch(self):
        """Parsed, not grepped - the docstring legitimately names the old
        import while explaining why it is gone."""
        import ast
        import pathlib
        tree = ast.parse(pathlib.Path(stitch.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "moe_stitch", (
                        "moe_stitch was never a PyPI package - it lived beside "
                        "fraunkenstein_universal.py, so this import could only "
                        "ever succeed with the Lab dir on sys.path")


class TestVerifyStitchIsRealVerification:
    """It used to read three config keys and call that verification."""

    def _write_cfg(self, d, **extra):
        d.mkdir(parents=True, exist_ok=True)
        body = {"num_experts": 2, "num_experts_per_tok": 2,
                "moe_intermediate_size": 128, "num_hidden_layers": 2,
                "expert_names": ["a", "b"]}
        body.update(extra)
        (d / "config.json").write_text(json.dumps(body), encoding="utf-8")
        return d

    def test_a_valid_config_alone_is_not_enough(self, tmp_path):
        """THE regression. A structurally perfect config.json with no weights
        on disk used to return True and print 'stitch OK'."""
        moe = self._write_cfg(tmp_path / "moe_untrained")
        assert stitch.verify_stitch(str(moe), output_root=str(tmp_path)) is False

    def test_missing_specialists_are_reported_not_ignored(self, tmp_path):
        moe = self._write_cfg(tmp_path / "moe_untrained")
        # no specialist_a / specialist_b directories exist
        assert stitch.verify_stitch(str(moe), output_root=str(tmp_path)) is False

    def test_missing_expert_names_fails(self, tmp_path):
        moe = tmp_path / "moe_untrained"
        moe.mkdir()
        (moe / "config.json").write_text(json.dumps(
            {"num_experts": 2, "num_experts_per_tok": 2,
             "moe_intermediate_size": 128}), encoding="utf-8")
        assert stitch.verify_stitch(str(moe), output_root=str(tmp_path)) is False

    def test_missing_config_fails(self, tmp_path):
        empty = tmp_path / "moe_untrained"
        empty.mkdir()
        assert stitch.verify_stitch(str(empty), output_root=str(tmp_path)) is False

    def test_unverifiable_is_false_not_true(self, tmp_path):
        """Without torch the tensor check cannot run. That is FALSE."""
        moe = self._write_cfg(tmp_path / "moe_untrained")
        for n in ("a", "b"):
            (tmp_path / st.FINETUNE_ARTIFACT.format(expert=n)).mkdir()
        # No safetensors anywhere, and possibly no torch: either way, not a pass.
        assert stitch.verify_stitch(str(moe), output_root=str(tmp_path)) is False

    def test_it_actually_opens_tensors(self):
        """Guard against a future 'simplification' back to key-checking."""
        import inspect
        src = inspect.getsource(stitch.verify_stitch)
        assert "torch.equal" in src, (
            "verification that never compares a tensor cannot tell a correct "
            "stitch from N identical copies of the anchor")


class TestArtifactNamesComeFromStages:
    """One place for the names, so a rename cannot half-land."""

    def test_specialist_dir_uses_the_stages_template(self, tmp_path):
        got = stitch._specialist_dir(str(tmp_path), "python")
        assert got.endswith(st.FINETUNE_ARTIFACT.format(expert="python"))

    def test_stitch_dir_uses_the_stages_constant(self):
        cfg = types.SimpleNamespace(output_root="/tmp/run", force=False)
        assert stitch.stitch_dir(cfg).endswith(st.ARTIFACTS[st.STITCH])

    def test_no_hardcoded_legacy_names(self):
        import inspect
        src = inspect.getsource(stitch)
        code = "\n".join(l for l in src.splitlines()
                         if not l.strip().startswith("#"))
        for legacy in ("qwen_coder_", "fraunkenstein_moe_untrained",
                       "fraunkenstein_agent_final"):
            assert legacy not in code, f"{legacy} still hardcoded in stitch.py"


class TestSkipPredicate:
    def test_force_defeats_the_skip(self, tmp_path):
        (tmp_path / "moe_untrained").mkdir()
        (tmp_path / "moe_untrained" / "config.json").write_text("{}")
        forced = types.SimpleNamespace(output_root=str(tmp_path), force=True)
        lazy = types.SimpleNamespace(output_root=str(tmp_path), force=False)
        assert stitch.stitch_is_done(forced) is False
        assert stitch.stitch_is_done(lazy) is True

    def test_absent_artifact_is_not_done(self, tmp_path):
        cfg = types.SimpleNamespace(output_root=str(tmp_path), force=False)
        assert stitch.stitch_is_done(cfg) is False
