"""Tests for preflight — the stage that used to check nothing.

It printed the config stamp and reported done. Every failure it could have
caught in two seconds was discovered later: a missing llama.cpp at stage 6,
after every specialist had trained and the router had run.
"""
import os
import types

import pytest

from ms_moe_maker import preflight as pf
from ms_moe_maker.evalrecord import FAIL, PASS, UNMEASURABLE
from ms_moe_maker.preflight import WARN
from ms_moe_maker.recipe import parse


def _cfg(tmp_path, **over):
    base = dict(base="Qwen/Qwen2.5-Coder-0.5B", size="0.5B",
                data_root=str(tmp_path / "data"),
                output_root=str(tmp_path / "out"),
                llama_cpp_dir=str(tmp_path / "nope-llama"),
                expert_names=["a", "b"])
    base.update(over)
    return types.SimpleNamespace(**base)


def _recipe(**over):
    body = {"schema_version": 1, "name": "t",
            "experts": [
                {"name": "a", "source": {"kind": "hf", "repo": "o/d"}},
                {"name": "b", "source": {"kind": "hf", "repo": "o/e"}},
            ]}
    body.update(over)
    rec, _ = parse(body)
    return rec


class TestSeverities:
    """A warning that stops the build is a failure wearing a friendly word."""

    def test_missing_exporter_is_a_warning_not_a_failure(self, tmp_path):
        """No llama.cpp means no GGUF — but the HF checkpoint is a real
        result, so this must not stop a six-hour build."""
        out = pf.Preflight()
        pf._check_exporter(out, _cfg(tmp_path))
        assert [c.status for c in out.checks] == [WARN]
        assert out.ok, "a missing exporter must not block the build"

    def test_a_warning_carries_a_remedy(self, tmp_path):
        out = pf.Preflight()
        pf._check_exporter(out, _cfg(tmp_path))
        assert "llama.cpp" in out.checks[0].remedy

    def test_blocking_is_only_true_for_fail(self):
        assert pf.Check("x", FAIL).blocking is True
        assert pf.Check("x", WARN).blocking is False
        assert pf.Check("x", UNMEASURABLE).blocking is False
        assert pf.Check("x", PASS).blocking is False


class TestSources:
    def test_a_missing_local_path_fails_before_anything_downloads(self, tmp_path):
        rec = _recipe(experts=[
            {"name": "a", "source": {"kind": "local",
                                     "path": str(tmp_path / "gone")}},
            {"name": "b", "source": {"kind": "hf", "repo": "o/d"}},
        ])
        out = pf.Preflight()
        pf._check_sources(out, rec)
        assert not out.ok
        assert any("does not exist" in c.detail for c in out.failures)

    def test_an_existing_local_path_passes(self, tmp_path):
        d = tmp_path / "corpus"
        d.mkdir()
        rec = _recipe(experts=[
            {"name": "a", "source": {"kind": "local", "path": str(d)}},
            {"name": "b", "source": {"kind": "hf", "repo": "o/d"}},
        ])
        out = pf.Preflight()
        pf._check_sources(out, rec)
        assert out.ok

    def test_gh_sources_are_accepted(self, tmp_path):
        rec = _recipe(experts=[
            {"name": "a", "source": {"kind": "gh", "repo": "o/r"}},
            {"name": "b", "source": {"kind": "gh", "repo": "o/r2"}},
        ])
        out = pf.Preflight()
        pf._check_sources(out, rec)
        assert out.ok


class TestRootsHaveNoSideEffects:
    """--plan runs preflight, and --plan promises to run nothing."""

    def test_it_does_not_create_the_run_directories(self, tmp_path):
        cfg = _cfg(tmp_path)
        out = pf.Preflight()
        pf._check_roots(out, cfg)
        assert not os.path.exists(cfg.output_root), (
            "preflight created the run directory - `--plan` must not litter "
            "someone's home folder as the price of telling them what would "
            "happen")
        assert not os.path.exists(cfg.data_root)

    def test_it_leaves_no_probe_file_behind(self, tmp_path):
        """It should not write anything at all - not even briefly.

        The original wrote a dotfile and deleted it. On a bridged/network
        mount the create was denied outright and preflight reported a
        perfectly good disk as unwritable, blocking the build at stage one for
        a reason that was not true.
        """
        cfg = _cfg(tmp_path)
        out = pf.Preflight()
        pf._check_roots(out, cfg)
        assert list(tmp_path.rglob(".msmoe-write-probe")) == []
        assert list(tmp_path.rglob("*.probe")) == []

    def test_an_unwritable_root_is_a_failure(self, tmp_path):
        if os.geteuid() == 0:
            pytest.skip("root ignores the permission bits")
        locked = tmp_path / "locked"
        locked.mkdir()
        locked.chmod(0o500)
        try:
            out = pf.Preflight()
            pf._check_roots(out, _cfg(tmp_path, output_root=str(locked / "x"),
                                      data_root=str(locked / "y")))
            assert not out.ok
        finally:
            locked.chmod(0o700)


class TestDiskEstimate:
    def test_it_scales_with_experts_and_size(self):
        small = pf._estimated_gb(types.SimpleNamespace(size="0.5B",
                                                      expert_names=["a", "b"]))
        big = pf._estimated_gb(types.SimpleNamespace(size="32B",
                                                    expert_names=["a"] * 5))
        assert big > small

    def test_a_junk_size_does_not_crash(self):
        assert pf._estimated_gb(
            types.SimpleNamespace(size="banana", expert_names=[])) > 0


class TestRender:
    def test_failures_are_rendered_last(self, tmp_path):
        out = pf.Preflight()
        out.add("zzz-fail", FAIL, "broken", "fix it")
        out.add("aaa-pass", PASS, "fine")
        lines = pf.render(out)
        assert "aaa-pass" in lines[0]
        assert any("zzz-fail" in l for l in lines[-2:]), (
            "the thing to act on belongs nearest the prompt")

    def test_remedies_are_printed(self, tmp_path):
        out = pf.Preflight()
        out.add("x", FAIL, "broken", "do the thing")
        assert any("do the thing" in l for l in pf.render(out))
