"""Ask the cheap question before the expensive stage, for BOTH stacks.

THE ASYMMETRY THIS FILE PINS. llama.cpp has been checked since the start, and
as a WARNING, correctly: a missing exporter costs the GGUF and you still keep
the checkpoint. vLLM had no check anywhere - `from vllm import LLM` sat bare in
the teacher's constructor, so a box without it got a ModuleNotFoundError from
inside the synth stage, which on a real gauntlet is fifty minutes past
preflight, past abliterate, past corpus collection, on a booked GPU.

Making `use_vllm` reachable from a recipe widened that, so it is checked now.
"""
import sys
import types

import pytest

from ms_moe_maker.run import preflight as pf_mod
from ms_moe_maker.run.preflight import FAIL, PASS, Preflight


class _Config:
    """Only the fields _check_generator reads."""

    def __init__(self, **kw):
        self.use_vllm = kw.get("use_vllm", False)
        self.teacher_batch = kw.get("teacher_batch", 96)
        self.synth_experts = kw.get("synth_experts", [])
        self.tools_expert_name = kw.get("tools_expert_name", "")
        self.reasoning_expert_name = kw.get("reasoning_expert_name", "")
        self.reasoning_experts = kw.get("reasoning_experts", [])


def _check(monkeypatch, present, **kw):
    """Run the check with vllm's importability forced either way."""
    import importlib.util

    real = importlib.util.find_spec

    def fake(name, *a, **k):
        if name == "vllm":
            return object() if present else None
        return real(name, *a, **k)

    monkeypatch.setattr(importlib.util, "find_spec", fake)
    pf = Preflight()
    pf_mod._check_generator(pf, _Config(**kw))
    return pf


class TestItRefusesBeforeTheGpuIsBooked:

    def test_vllm_asked_for_and_missing_is_a_FAILURE(self, monkeypatch):
        pf = _check(monkeypatch, present=False, use_vllm=True,
                    tools_expert_name="agentcore")
        assert not pf.ok, (
            "the build would start, spend fifty minutes on abliterate and "
            "corpus collection, and then die on an import")
        bad = pf.failures[0]
        assert "vllm is not installed" in bad.detail
        assert "runtime.use_vllm" in bad.detail or "use_vllm" in bad.detail

    def test_the_remedy_names_both_ways_out(self, monkeypatch):
        """A refusal that does not say what to do instead is a wall."""
        bad = _check(monkeypatch, present=False, use_vllm=True,
                     synth_experts=["a"]).failures[0]
        assert "pip install vllm" in bad.remedy
        assert "use_vllm: false" in bad.remedy

    def test_it_says_why_it_will_not_just_fall_back(self, monkeypatch):
        """THE DESIGN DECISION, written where somebody hits it. Falling back
        would generate at batch 96 with a different sampler under a build_id
        whose fingerprint says vLLM - the artifact and its claim disagreeing
        is worse than a refusal."""
        bad = _check(monkeypatch, present=False, use_vllm=True,
                     synth_experts=["a"]).failures[0]
        assert "fingerprint" in bad.remedy
        assert "96" in bad.remedy and "512" in bad.remedy

    def test_vllm_asked_for_and_present_passes(self, monkeypatch):
        pf = _check(monkeypatch, present=True, use_vllm=True,
                    synth_experts=["a"], teacher_batch=512)
        assert pf.ok
        assert "vLLM" in pf.checks[0].detail and "512" in pf.checks[0].detail


class TestItDoesNotAskForWhatTheBuildWillNotUse:

    def test_a_recipe_with_no_synth_work_is_not_asked_for_a_teacher(
            self, monkeypatch):
        """No synth expert, no tools expert, no reasoning expert - the build
        never constructs a teacher, so a failure about its serving stack would
        be a preflight refusal about a stage that is not in the plan."""
        pf = _check(monkeypatch, present=False, use_vllm=True)
        assert pf.checks == [], pf.checks

    @pytest.mark.parametrize("field,value", [
        ("synth_experts", ["lore"]),
        ("tools_expert_name", "agentcore"),
        ("reasoning_expert_name", "deliberation"),
        ("reasoning_experts", ["csharp"]),
    ])
    def test_every_way_of_acquiring_synth_work_counts(self, monkeypatch,
                                                      field, value):
        """Four doors to a teacher. Missing one means a build that needs vllm
        sails through preflight and dies later - which is the bug."""
        pf = _check(monkeypatch, present=False, use_vllm=True,
                    **{field: value})
        assert not pf.ok, f"{field} did not count as needing a teacher"


class TestThePlainPathIsReportedToo:

    def test_it_names_the_generator_and_the_batch(self, monkeypatch):
        """'transformers, batch 96' is the answer to 'why is synth taking
        three hours', said where somebody is already looking."""
        pf = _check(monkeypatch, present=False, use_vllm=False,
                    synth_experts=["a"])
        assert pf.ok
        assert pf.checks[0].status == PASS
        assert "transformers" in pf.checks[0].detail
        assert "96" in pf.checks[0].detail

    def test_missing_vllm_is_irrelevant_when_it_was_not_asked_for(
            self, monkeypatch):
        assert _check(monkeypatch, present=False, use_vllm=False,
                      synth_experts=["a"]).ok


class TestTheGuardedImport:
    """Preflight refuses first, but somebody can get past it - a resumed run,
    a stage invoked directly. The bare import gave them a ModuleNotFoundError
    from inside a generator with nothing tying it to the recipe line."""

    def test_the_teacher_raises_a_sentence_not_a_traceback(self, monkeypatch):
        from ms_moe_maker.data.synth import _VLLMTeacher

        # No vllm on the path, whatever this box has.
        monkeypatch.setitem(sys.modules, "vllm", None)
        real_import = __builtins__["__import__"] if isinstance(
            __builtins__, dict) else __builtins__.__import__

        def blocked(name, *a, **k):
            if name == "vllm":
                raise ImportError("No module named 'vllm'")
            return real_import(name, *a, **k)

        monkeypatch.setattr("builtins.__import__", blocked)
        with pytest.raises(RuntimeError) as exc:
            _VLLMTeacher(types.SimpleNamespace())
        said = str(exc.value)
        assert "use_vllm" in said, said
        assert "pip install vllm" in said
        assert "fingerprint" in said, (
            "the refusal does not say why it will not quietly downgrade")
