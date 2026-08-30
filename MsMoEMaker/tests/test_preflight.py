"""Tests for preflight — the stage that used to check nothing.

It printed the config stamp and reported done. Every failure it could have
caught in two seconds was discovered later: a missing llama.cpp at stage 6,
after every specialist had trained and the router had run.
"""
import os
import types

import pytest

from ms_moe_maker.run import preflight as pf
from ms_moe_maker.eval.record import FAIL, PASS, UNMEASURABLE
from ms_moe_maker.run.preflight import WARN
from ms_moe_maker.config.recipe import parse


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


def test_a_non_qwen_base_is_refused_even_from_env():
    """validate() only checks recipe.base; an env override or a box `models:`
    entry swaps in the REAL base unseen. Preflight sees the resolved id, so it
    refuses before anything expensive starts - the failure the recipe check's
    own comment exists to prevent, asked of the right id."""
    out = pf.Preflight()
    pf._check_base_model(
        out,
        types.SimpleNamespace(base="meta-llama/Llama-3.1-8B-Instruct"),
        offline=True)
    assert any(c.status == FAIL for c in out.checks), out.checks
    assert "not a supported MoE architecture" in out.checks[0].detail


def test_a_qwen_base_still_passes_the_arch_check():
    out = pf.Preflight()
    pf._check_base_model(
        out,
        types.SimpleNamespace(base="Qwen/Qwen2.5-Coder-0.5B-Instruct"),
        offline=True)
    assert all(c.status != FAIL for c in out.checks), out.checks


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


class TestDatasetReachability:
    """The check that was missing when a real run died at stage 1.

    Preflight confirmed the base model was reachable, reported everything
    green, and then corpus collection went looking for a dataset repo and
    failed there. Same class of rot, same two-second check, not applied.
    """

    def _recipe(self, experts):
        from ms_moe_maker.config.recipe import parse
        rec, _ = parse({"schema_version": 1, "name": "t", "experts": experts})
        return rec

    def test_offline_reports_unmeasurable_not_pass(self):
        """Skipping a check is not the same as passing it."""
        out = pf.Preflight()
        pf._check_datasets(out, self._recipe([
            {"name": "a", "source": {"kind": "hf", "repo": "o/d"}},
            {"name": "b", "source": {"kind": "hf", "repo": "o/e"}}]),
            offline=True)
        assert [c.status for c in out.checks] == [UNMEASURABLE]
        assert out.ok, "an unchecked dataset must not block the build"

    def _fake_hub(self, monkeypatch, api):
        """Inject a stand-in huggingface_hub.

        monkeypatch.setattr on the real module is not an option: these tests
        have to pass on the BASE install, where huggingface_hub is not present
        at all - which is the same laptop promise the package makes. Stubbing
        sys.modules tests our code rather than the hub client.
        """
        import sys
        import types
        mod = types.ModuleType("huggingface_hub")
        mod.HfApi = lambda: api
        monkeypatch.setitem(sys.modules, "huggingface_hub", mod)

    def test_a_dead_repo_is_a_failure_with_a_remedy(self, monkeypatch):
        import ms_moe_maker.run.preflight as mod

        class _Api:
            def dataset_info(self, repo):
                raise RuntimeError("404 Not Found")

        self._fake_hub(monkeypatch, _Api())
        out = pf.Preflight()
        mod._check_datasets(out, self._recipe([
            {"name": "a", "source": {"kind": "hf", "repo": "nope/gone"}},
            {"name": "b", "source": {"kind": "hf", "repo": "nope/gone2"}}]))
        assert not out.ok
        assert all(c.remedy for c in out.failures), (
            "a dead repo id must come with what to do about it")

    def test_stack_sources_check_the_corpus_repo(self, monkeypatch):
        """A `stack` expert names a language, not a repo - but the scan still
        pulls from one, so that is what gets checked."""
        import ms_moe_maker.run.preflight as mod
        from ms_moe_maker.data.synth import STACK_REPO
        seen = []

        class _Api:
            def dataset_info(self, repo):
                seen.append(repo)

        self._fake_hub(monkeypatch, _Api())
        out = pf.Preflight()
        mod._check_datasets(out, self._recipe([
            {"name": "python", "source": {"kind": "stack", "language": "Python"}},
            {"name": "csharp", "source": {"kind": "stack", "language": "C#"}}]))
        assert seen == [STACK_REPO], seen
        assert out.ok

    def test_one_request_per_repo_not_per_expert(self, monkeypatch):
        """Four stack experts is still one corpus."""
        import ms_moe_maker.run.preflight as mod
        seen = []

        class _Api:
            def dataset_info(self, repo):
                seen.append(repo)

        self._fake_hub(monkeypatch, _Api())
        out = pf.Preflight()
        mod._check_datasets(out, self._recipe([
            {"name": n, "source": {"kind": "stack", "language": n}}
            for n in ("python", "csharp", "shell", "powershell")]))
        assert len(seen) == 1, seen

    def test_the_checked_id_is_the_one_data_py_requests(self):
        """A reachability check against a merely-similar id is worse than none,
        because it passes."""
        import inspect
        from ms_moe_maker.data import synth as data
        src = inspect.getsource(data._collect_from_shards)
        assert "STACK_REPO" in src
        assert data.STACK_REPO.count("/") == 1

    def test_local_sources_need_no_network(self, monkeypatch):
        import ms_moe_maker.run.preflight as mod
        calls = []

        class _Api:
            def dataset_info(self, repo):
                calls.append(repo)

        self._fake_hub(monkeypatch, _Api())
        out = pf.Preflight()
        mod._check_datasets(out, self._recipe([
            {"name": "a", "source": {"kind": "local", "path": "/tmp"}},
            {"name": "b", "source": {"kind": "local", "path": "/tmp"}}]))
        assert calls == [], "a local corpus must not touch the network"
        assert out.ok

    def test_a_missing_hub_is_unmeasurable_not_a_failure(self, monkeypatch):
        """On a base install the client is absent. That blocks nothing."""
        import sys
        import ms_moe_maker.run.preflight as mod
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        out = pf.Preflight()
        mod._check_datasets(out, self._recipe([
            {"name": "a", "source": {"kind": "hf", "repo": "o/d"}},
            {"name": "b", "source": {"kind": "hf", "repo": "o/e"}}]))
        assert out.ok
