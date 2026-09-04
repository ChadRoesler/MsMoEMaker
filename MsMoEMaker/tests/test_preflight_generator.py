"""Ask the cheap question before the expensive stage, for BOTH stacks.

THE ASYMMETRY THIS FILE PINS. llama.cpp has been checked since the start, and
as a WARNING, correctly: a missing exporter costs the GGUF and you still keep
the checkpoint. vLLM had no check anywhere - `from vllm import LLM` sat bare in
the teacher's constructor, so a box without it got a ModuleNotFoundError from
inside the synth stage, which on a real gauntlet is fifty minutes past
preflight, past abliterate, past corpus collection, on a booked GPU.

Making `use_vllm` reachable from a recipe widened that, so it is checked now.
"""
import os
import sys
import types

import pytest

from ms_moe_maker.run import preflight as pf_mod
from ms_moe_maker.run.preflight import FAIL, PASS, WARN, Preflight


class _Config:
    """Only the fields _check_generator reads."""

    def __init__(self, **kw):
        self.use_vllm = kw.get("use_vllm", False)
        self.teacher_batch = kw.get("teacher_batch", 96)
        self.vllm_batch = kw.get("vllm_batch", 512)
        self.synth_experts = kw.get("synth_experts", [])
        self.tools_expert_name = kw.get("tools_expert_name", "")
        self.reasoning_expert_name = kw.get("reasoning_expert_name", "")
        self.reasoning_experts = kw.get("reasoning_experts", [])
        self.use_unsloth = kw.get("use_unsloth", False)
        self.load_in_4bit = kw.get("load_in_4bit", False)
        self.optim = kw.get("optim", "adamw_torch")


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
                    synth_experts=["a"])
        assert pf.ok
        assert "vLLM" in pf.checks[0].detail and "512" in pf.checks[0].detail

    def test_each_path_reports_the_batch_field_it_actually_reads(
            self, monkeypatch):
        """TWO BATCH FIELDS, ONE READ PER PATH. _HFTeacher takes
        config.teacher_batch and _VLLMTeacher takes config.vllm_batch - and
        both say 512 under vLLM today, so reading the wrong one prints the
        right number for the wrong reason and diverges the first time somebody
        tunes one of them."""
        vllm = _check(monkeypatch, present=True, use_vllm=True,
                      synth_experts=["a"], teacher_batch=7, vllm_batch=99)
        assert "99" in vllm.checks[0].detail, vllm.checks[0].detail
        assert "7" not in vllm.checks[0].detail

        plain = _check(monkeypatch, present=False, use_vllm=False,
                       synth_experts=["a"], teacher_batch=7, vllm_batch=99)
        assert "7" in plain.checks[0].detail, plain.checks[0].detail
        assert "99" not in plain.checks[0].detail


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


def _trainer(monkeypatch, present, **kw):
    import importlib.util

    real = importlib.util.find_spec

    def fake(name, *a, **k):
        if name == "unsloth":
            return object() if present else None
        return real(name, *a, **k)

    monkeypatch.setattr(importlib.util, "find_spec", fake)
    pf = Preflight()
    pf_mod._check_trainer(pf, _Config(**kw))
    return pf


class TestUnslothFallsBackButSaysSo:
    """FALL BACK AND SAY SO LOUDLY, AT VALIDATE TIME - the llama.cpp shape,
    not the vLLM one, because a plain fine-tune is a real result.

    What it replaces was neither: a one-line print five hours into a run, and
    an `if want_unsloth and not unsloth_available: raise` that could never fire
    because the except above it had already set want_unsloth to False. Dead
    code standing in for a guard nobody had.
    """

    def test_missing_unsloth_warns_and_does_not_block(self, monkeypatch):
        pf = _trainer(monkeypatch, present=False, use_unsloth=True,
                      optim="adamw_8bit")
        assert pf.ok, "a plain fine-tune is a real result and must not refuse"
        assert pf.warnings and pf.warnings[0].status == WARN
        assert "plain path" in pf.warnings[0].detail

    def test_the_warning_names_the_optimiser_that_comes_along_for_the_ride(
            self, monkeypatch):
        """THE HALF THE OLD PRINT MISSED. config.optim was resolved to
        adamw_8bit from use_unsloth back in build_config, and finetune passes
        optim=config.optim whichever path it took - so the plain trainer gets
        the 8-bit optimiser and needs bitsandbytes to honour it."""
        warn = _trainer(monkeypatch, present=False, use_unsloth=True,
                        optim="adamw_8bit").warnings[0]
        assert "adamw_8bit" in warn.remedy
        assert "bitsandbytes" in warn.remedy

    def test_the_warning_says_the_manifest_will_be_wrong(self, monkeypatch):
        """use_unsloth is in the build fingerprint, so a silent fallback
        leaves the manifest describing a run that did not happen."""
        warn = _trainer(monkeypatch, present=False, use_unsloth=True).warnings[0]
        assert "manifest" in warn.remedy

    def test_unsloth_present_is_a_quiet_pass(self, monkeypatch):
        pf = _trainer(monkeypatch, present=True, use_unsloth=True,
                      optim="adamw_8bit")
        assert pf.ok and not pf.warnings
        assert "unsloth" in pf.checks[0].detail

    def test_nobody_asked_means_nothing_is_said(self, monkeypatch):
        assert _trainer(monkeypatch, present=False).checks == []


class TestFourBitIsTheOneHardRefusal:
    """finetune already raises for this before training, deliberately. Saying
    it at preflight costs nothing and says it on the laptop instead of on the
    build box after the recipe has loaded."""

    def test_four_bit_without_unsloth_fails(self, monkeypatch):
        pf = _trainer(monkeypatch, present=False, load_in_4bit=True)
        assert not pf.ok
        assert "load_in_4bit" in pf.failures[0].detail
        assert "save_pretrained_merged" in pf.failures[0].remedy

    def test_four_bit_with_unsloth_is_fine(self, monkeypatch):
        assert _trainer(monkeypatch, present=True, load_in_4bit=True,
                        use_unsloth=True).ok

    def test_the_refusal_beats_the_warning(self, monkeypatch):
        """Both conditions at once is still one answer, and it is the FAIL:
        4-bit would train for hours and then die at the save."""
        pf = _trainer(monkeypatch, present=False, load_in_4bit=True,
                      use_unsloth=True)
        assert len(pf.checks) == 1 and pf.checks[0].status == FAIL


class TestTheEngineIsBuiltOnce:
    """It used to construct the LLM, then - if vllm_quantization was set -
    construct a SECOND one over the top. At gpu_memory_utilization 0.88 the
    discarded engine has already claimed almost the whole card, so the
    quantized path was a double load into a GPU with no room left.

    Dormant, because vllm_quantization defaults to None. That is how it
    survived: the only way to reach it is to set the knob, and setting the knob
    is when you can least afford an OOM.
    """

    def test_only_one_engine_is_constructed(self):
        import inspect

        from ms_moe_maker.data import synth

        src = inspect.getsource(synth._VLLMTeacher.__init__)
        assert src.count("LLM(**kwargs)") == 1
        assert "self.llm = LLM(" not in src.replace("self.llm = LLM(**kwargs)",
                                                    ""), (
            "a second engine construction is back")

    def test_quantization_is_a_kwarg_not_a_second_engine(self):
        import inspect

        from ms_moe_maker.data import synth

        src = inspect.getsource(synth._VLLMTeacher.__init__)
        assert 'kwargs["quantization"]' in src

    def test_the_engine_kwargs_use_the_names_vllm_actually_has(self):
        """cache_dir is an HF parameter, not a vLLM one.

        Every other `cache_dir=config.hf_home` in synth.py goes to a real HF
        API - AutoTokenizer.from_pretrained, load_dataset - where it is
        correct. This constructor forwards **kwargs to EngineArgs, which has
        `download_dir` and no `cache_dir`, so the copy-pasted idiom would have
        TypeError'd on the first vLLM build anyone ran. Checked against the
        installed engine, not remembered.
        """
        import inspect

        from ms_moe_maker.data import synth

        src = inspect.getsource(synth._VLLMTeacher.__init__)
        body = src[src.index("kwargs = dict("):src.index("SamplingParams(")]
        code = "\n".join(l for l in body.splitlines()
                         if not l.strip().startswith("#"))
        assert "download_dir=config.hf_home" in code
        assert "cache_dir=" not in code, (
            "cache_dir is back in the engine kwargs - vLLM will reject it")

    def test_the_tokenizer_keeps_its_own_cache_dir(self):
        """And NOT by deleting it everywhere: from_pretrained's cache_dir is
        correct, and it is what makes the tokenizer and the weights share one
        directory instead of downloading the teacher twice."""
        import inspect

        from ms_moe_maker.data import synth

        src = inspect.getsource(synth._VLLMTeacher.__init__)
        head = src[:src.index("kwargs = dict(")]
        assert "cache_dir=config.hf_home" in head

    def test_a_rejected_engine_argument_names_itself(self, monkeypatch):
        """The kwargs are pinned to vLLM 0.28. 0.29 gets to disagree, and the
        person who meets that deserves one sentence rather than a TypeError
        from inside a generator forty minutes into a build."""
        import sys
        import types

        # SKIPPED NAMING WHAT IT WAITS FOR. The teacher loads a tokenizer
        # before it builds the engine, so this path genuinely needs
        # transformers - it is not here on a viewer-only box, and a silent
        # skip would hide that this assertion never runs anywhere.
        pytest.importorskip(
            "transformers",
            reason="the vLLM teacher loads a tokenizer before the engine, so "
                   "this path needs transformers installed")

        from ms_moe_maker.data import synth

        fake = types.ModuleType("vllm")

        class _LLM:
            def __init__(self, **kw):
                raise TypeError("unexpected keyword argument 'cache_dir'")

        fake.LLM = _LLM
        fake.SamplingParams = lambda **kw: None
        monkeypatch.setitem(sys.modules, "vllm", fake)

        cfg = types.SimpleNamespace(
            teacher_model="x/y", hf_home="/tmp/hf", vllm_gpu_util=0.88,
            vllm_max_len=4096, vllm_quantization=None, vllm_batch=512,
            teacher_max_new=512, seed=42)
        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained",
            lambda *a, **k: types.SimpleNamespace(eos_token="</s>"),
            raising=False)

        with pytest.raises(RuntimeError) as exc:
            synth._VLLMTeacher(cfg)
        said = str(exc.value)
        assert "cache_dir" in said, said
        assert "EngineArgs" in said, "the message does not say where to look"


class TestTheForkGuard:
    """vLLM v1 runs its EngineCore in a separate process, and on Linux that
    forks - which cannot re-initialise a CUDA context the parent already holds.

    vLLM detects this itself and overrides to spawn. On a real build here it
    DIDN'T: the engine forked and the stage died, while the same box with an
    explicit `torch.zeros(1, device="cuda")` first tripped the detector fine.
    The likely gap is preflight's `torch.cuda.is_available()`, which touches
    the driver enough to poison a fork without making `is_initialized()` true.

    This pipeline has always touched CUDA by the time it builds a teacher, so
    it says so rather than waiting on a detector that already missed once.
    """

    def _fake_vllm(self, monkeypatch, record):
        import sys
        import types

        vllm = types.ModuleType("vllm")

        class _LLM:
            def __init__(self, **kw):
                record["env_at_construct"] = os.environ.get(
                    "VLLM_WORKER_MULTIPROC_METHOD")

        vllm.LLM = _LLM
        vllm.SamplingParams = lambda **kw: types.SimpleNamespace(**kw)
        monkeypatch.setitem(sys.modules, "vllm", vllm)

        tf = types.ModuleType("transformers")
        tf.AutoTokenizer = types.SimpleNamespace(
            from_pretrained=lambda *a, **k: types.SimpleNamespace(
                eos_token="</s>"))
        monkeypatch.setitem(sys.modules, "transformers", tf)

    def _cfg(self):
        import types

        return types.SimpleNamespace(
            teacher_model="x/y", hf_home="/tmp/hf", vllm_gpu_util=0.88,
            vllm_max_len=4096, vllm_quantization=None, vllm_batch=512,
            teacher_max_new=512, seed=42)

    def test_the_start_method_is_spawn_by_the_time_the_engine_is_built(
            self, monkeypatch):
        from ms_moe_maker.data import synth

        monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
        record = {}
        self._fake_vllm(monkeypatch, record)
        synth._VLLMTeacher(self._cfg())
        assert record["env_at_construct"] == "spawn", (
            "the engine was built with the default start method - on Linux "
            "that is fork, and the parent already holds a CUDA context")

    def test_an_operator_who_set_it_themselves_still_wins(self, monkeypatch):
        """setdefault, not assignment. Somebody debugging with fork on purpose
        should not be quietly overruled by a library call."""
        from ms_moe_maker.data import synth

        monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "forkserver")
        record = {}
        self._fake_vllm(monkeypatch, record)
        synth._VLLMTeacher(self._cfg())
        assert record["env_at_construct"] == "forkserver"

    def test_it_is_set_BEFORE_the_import_not_after(self):
        """THE ORDERING IS THE WHOLE THING, and it is invisible once it works.

        The multiprocessing context is settled as vllm loads, so a tidy-up
        that hoists the import above the setdefault puts the fork crash back
        with nothing on screen to say why. A source-order assertion is crude,
        and it is exactly the regression worth catching.
        """
        import inspect

        from ms_moe_maker.data import synth

        # COMMENTS STRIPPED FIRST. The comment block above the import quotes
        # `from vllm import LLM` verbatim while explaining it, so a raw search
        # finds the prose before the code and the assertion compares two
        # things that are not the two lines under test.
        src = inspect.getsource(synth._VLLMTeacher.__init__)
        code = "\n".join(l for l in src.splitlines()
                         if not l.strip().startswith("#"))
        env_line = code.index('setdefault("VLLM_WORKER_MULTIPROC_METHOD"')
        import_line = code.index("from vllm import LLM")
        assert env_line < import_line, (
            "VLLM_WORKER_MULTIPROC_METHOD is set after vllm is imported, "
            "which is too late for the multiprocessing context")


# ── where the think tags come from ──────────────────────────────────────────

def _reasoning_cfg(**kw):
    return types.SimpleNamespace(
        reasoning_open=kw.get("open", "<think>"),
        reasoning_close=kw.get("close", "</think>"),
        reasoning_type=kw.get("type", "xml"),
        reasoning_interwoven=kw.get("interwoven", False),
        reasoning_teacher=kw.get("teacher",
                                 "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
        teacher_model=kw.get("teacher_model", ""),
        reasoning_expert_name=kw.get("expert", "deliberation"),
        reasoning_experts=kw.get("experts", []))


def _reasoning_check(cfg, recipe=None):
    pf = Preflight()
    pf_mod._check_reasoning(pf, cfg, recipe or types.SimpleNamespace(
        base="Qwen/Qwen2.5-Coder-0.5B-Instruct", base_kind="nonreasoning"))
    return pf


class TestWhereTheTagsCameFrom:
    """A wrong tag style is a SILENT wrong answer - no delimiters found, eval
    reports "did not reason", the think block gets scored as the answer. Three
    layers merge into the table that decides it, so the answer to "where did
    these come from" belongs in the preflight somebody already ran.
    """

    def test_it_names_the_tags_the_specialist_will_learn(self):
        pf = _reasoning_check(_reasoning_cfg())
        assert pf.ok
        detail = " ".join(c.detail for c in pf.checks)
        assert "<think>…</think>" in detail

    def test_it_names_the_family_the_hint_and_the_FILE(self):
        """The file is the point. 'it matched xml' still leaves you grepping."""
        pf = _reasoning_check(_reasoning_cfg())
        detail = " ".join(c.detail for c in pf.checks)
        assert "R1-Distill-Qwen" in detail, detail
        assert "r1distillqwen" in detail, "the hint that won is not named"
        assert "reasoning.yaml" in detail, "nothing says which file said so"

    def test_a_nonreasoning_BASE_does_not_gag_the_teacher_line(self):
        """THE BUG THIS CHECK SHIPPED WITH, caught by its own first run.

        It asked explain_base about the BASE with the BASE's kind, and printed
        "base_kind is nonreasoning, so nothing reasons" on the same line as
        <think>…</think>. Which is the normal case, not an edge one: the base
        is nonreasoning precisely BECAUSE you are teaching it to reason, from
        traces a separate R1 teacher writes.
        """
        pf = _reasoning_check(_reasoning_cfg())
        detail = " ".join(c.detail for c in pf.checks)
        assert "nothing reasons" not in detail, detail
        assert "DeepSeek-R1-Distill-Qwen-7B" in detail

    def test_a_teacher_that_speaks_a_different_dialect_is_stated_not_warned(
            self):
        """DeepSeek-R1 proper emits <|reasoning|> while the specialist learns
        <think>. That is correct and by design - synth says conflating the two
        "rejected a good teacher's output 288 times in a row"."""
        pf = _reasoning_check(_reasoning_cfg(teacher="deepseek-ai/DeepSeek-R1"))
        assert pf.ok, "a legitimately different teacher dialect was flagged"
        detail = " ".join(c.detail for c in pf.checks)
        assert "<|reasoning|>" in detail
        assert "same as the target" not in detail

    def test_the_same_dialect_says_so(self):
        detail = " ".join(c.detail for c in _reasoning_check(
            _reasoning_cfg()).checks)
        assert "same as the target" in detail

    def test_an_unknown_teacher_warns_rather_than_assuming_quietly(self):
        """No family match means synth assumes the teacher speaks the target
        style. Right often enough to do; wrong quietly enough to say."""
        pf = _reasoning_check(_reasoning_cfg(teacher="mystery/Model-9000"))
        assert pf.ok, "an unknown teacher is a warning, not a refusal"
        warn = [c for c in pf.checks if c.status == WARN]
        assert warn, [c.detail for c in pf.checks]
        assert "matches no family" in warn[0].detail
        assert "reasoning.yaml" in warn[0].remedy

    def test_no_delimiters_at_all_warns(self):
        pf = _reasoning_check(_reasoning_cfg(open="", close=""))
        warn = [c for c in pf.checks if c.status == WARN]
        assert warn and "no delimiters resolved" in warn[0].detail

    def test_a_build_with_no_reasoning_work_says_nothing(self):
        """Silence is the right answer for a recipe that never reasons."""
        pf = _reasoning_check(_reasoning_cfg(open="", close="", expert="",
                                             experts=[]))
        assert pf.checks == [], [c.detail for c in pf.checks]

    def test_a_broken_user_table_is_surfaced(self, monkeypatch):
        """A malformed ~/.msmoe/reasoning.yaml degrades to the packaged table
        rather than raising, so the only way anyone finds out is if something
        asks."""
        monkeypatch.setattr(pf_mod, "_check_reasoning",
                            pf_mod._check_reasoning)  # keep the real one
        from ms_moe_maker.config import reasoning as rz

        monkeypatch.setattr(rz, "load_errors",
                            lambda: ["~/.msmoe/reasoning.yaml: unreadable"])
        pf = _reasoning_check(_reasoning_cfg())
        assert any("unreadable" in c.detail for c in pf.checks), (
            "the table fell back a layer and preflight said nothing")


class TestExplainBaseCarriesItsReceipts:
    """style_for_base returns the key alone - all a build needs, and exactly
    nothing when somebody is staring at a trace wondering where its delimiters
    came from."""

    def test_the_longest_hint_wins_and_says_which(self):
        from ms_moe_maker.config.reasoning import explain_base, load

        styles, fams, _ = load(include_user=False)
        why = explain_base("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                           "auto", styles, fams)
        assert why["style"] == "xml"
        assert why["hint"] == "r1distillqwen", (
            "the DeepSeek family's shorter hint won - an R1-distill would be "
            "told to emit <|reasoning|>, which it does not")
        assert why["family"] == "R1-Distill-Qwen"
        assert why["source"].endswith("reasoning.yaml")

    def test_the_longest_hint_wins_REGARDLESS_OF_ORDER(self):
        """The claim is order-independence, and the packaged table cannot test
        it: its long-hint family happens to come last, so last-write-wins gives
        the same answer as longest-wins and a mutation that breaks the rule
        sails through. So this builds the table the other way round.
        """
        from ms_moe_maker.config.reasoning import (ReasoningFamily,
                                                   ReasoningStyle,
                                                   explain_base)

        styles = {"long": ReasoningStyle("Long", "<a>", "</a>", source="L"),
                  "short": ReasoningStyle("Short", "<b>", "</b>", source="S")}
        # The SHORT hint is declared last, so anything order-dependent picks it.
        fams = {
            "long": ReasoningFamily("Long Family", ("r1distillqwen",), "long",
                                    source="L"),
            "short": ReasoningFamily("Short Family", ("deepseek",), "short",
                                     source="S"),
        }
        why = explain_base("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                           "auto", styles, fams)
        assert why["hint"] == "r1distillqwen", why
        assert why["family"] == "Long Family"

    def test_the_source_is_the_file_that_defined_the_FAMILY(self):
        """Two files can be involved: one defines the matching rule, one
        defines the tags. "Where did this come from" means which file DECIDED,
        not which file spelled <think>."""
        from ms_moe_maker.config.reasoning import (ReasoningFamily,
                                                   ReasoningStyle,
                                                   explain_base)

        styles = {"xml": ReasoningStyle("XML", "<think>", "</think>",
                                        source="/packaged.yaml")}
        fams = {"mine": ReasoningFamily("Mine", ("model9000",), "xml",
                                        source="/home/me/.msmoe/reasoning.yaml")}
        why = explain_base("mystery/Model-9000", "auto", styles, fams)
        assert why["source"] == "/home/me/.msmoe/reasoning.yaml", (
            "it pointed at the packaged table for a rule that came from the "
            "operator's own file - which is the exact hunt this exists to end")
        assert why["style_source"] == "/packaged.yaml"

    def test_style_for_base_is_the_same_answer(self):
        """One derivation. The key a build acts on and the story a person is
        shown must not be two lookups that can disagree."""
        from ms_moe_maker.config.reasoning import (explain_base, load,
                                                   style_for_base)

        styles, fams, _ = load(include_user=False)
        for model in ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                      "deepseek-ai/DeepSeek-R1", "Qwen/QwQ-32B",
                      "mistralai/Mistral-7B"):
            for kind in ("auto", "reasoning", "nonreasoning"):
                assert (style_for_base(model, kind, styles, fams)
                        == explain_base(model, kind, styles, fams)["style"])

    def test_a_user_layer_is_named_as_the_source(self, tmp_path, monkeypatch):
        """THE WHOLE POINT: when you drop a family in ~/.msmoe/reasoning.yaml,
        the check has to say THAT file and not the packaged one."""
        from ms_moe_maker.config import reasoning as rz

        # ONLY A FAMILY, pointing at a style the PACKAGED table already has.
        # That is what somebody actually writes - "my model speaks <think>,
        # here is how to spot it" - and it is the case where the family's own
        # provenance is the only thing that can name the right file. An earlier
        # version of this fixture defined a style too, so the style's source
        # masked the family's and a loader that recorded nothing still passed.
        table = tmp_path / "mine.yaml"
        table.write_text(
            "Families:\n"
            "  - Key: mystery\n"
            "    FamilyName: Mystery Family\n"
            "    Models: [Model-9000]\n"
            "    PreferredStyle: xml\n", encoding="utf-8")
        monkeypatch.setenv(rz.USER_ENV, str(table))
        styles, fams, warns = rz.load()
        assert not warns, warns
        why = rz.explain_base("mystery/Model-9000", "auto", styles, fams)
        assert why["style"] == "xml"
        assert why["family"] == "Mystery Family"
        assert why["source"] == str(table), (
            "it pointed at the packaged table for a family that came from the "
            "operator's own file - the exact hunt this exists to end")
        assert why["style_source"].endswith("assets/reasoning.yaml"), (
            "the tags really did come from the packaged table; both halves "
            "should be reported honestly")
