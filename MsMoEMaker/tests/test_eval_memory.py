"""Eval must not eat the machine, and must not die on it either.

Two OOMs on a 128 GB unified-memory Spark produced, between them, zero
attributable evidence: the kernel killed the process, and the CUDA allocator
reported "free: 7906271232" of 130 GB total without anything in our code ever
having said how much it was holding. On a box where host and device share one
pool, a corpus read into a Python list IS CUDA memory.

These pin the class of bug, not the instance:

  * no reader in eval.py may materialise a corpus
  * an OOM is a measurement outcome (UNMEASURABLE), never a crash
"""
import ast
import json
import sys
import types
from pathlib import Path

import pytest

from ms_moe_maker import eval as ev

EVAL_SRC = Path(ev.__file__).read_text(encoding="utf-8")


# ── the corpus must never be materialised ─────────────────────────────────

def _read_text_callers(source: str):
    """Functions in `source` that call .read_text(), docstrings excluded.

    Docstrings are stripped because this guard has been tripped twice before
    by test and module PROSE describing the very pattern it forbids. A guard
    that matches its own explanation of itself is not a guard.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            node.value.value = ""
    out = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr in ("read_text", "readlines"):
                out.setdefault(fn.name, 0)
                out[fn.name] += 1
    return out


def test_no_corpus_is_read_whole():
    callers = _read_text_callers(EVAL_SRC)
    # eval_from_manifest reads the small JSON eval record a previous run
    # emitted. That is a result document, not a corpus, and it is the only
    # exemption.
    assert set(callers) <= {"eval_from_manifest"}, (
        f"whole-file reads reappeared in eval.py: {callers}. Use _iter_jsonl "
        f"or _reservoir - on unified memory this is CUDA's pool.")


def test_iter_jsonl_skips_blanks_and_survives_a_missing_file(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"text":"a"}\n\n   \n{"text":"b"}\n', encoding="utf-8")
    assert list(ev._iter_jsonl(str(p))) == ['{"text":"a"}', '{"text":"b"}']
    assert list(ev._iter_jsonl(str(tmp_path / "nope.jsonl"))) == []


def test_reservoir_draws_n_real_lines(tmp_path):
    p = tmp_path / "c.jsonl"
    lines = [json.dumps({"text": f"row {i}"}) for i in range(500)]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    picked = ev._reservoir(str(p), 20)
    assert len(picked) == 20
    assert len(set(picked)) == 20, "reservoir returned a duplicate"
    assert set(picked) <= set(lines)
    assert ev._reservoir(str(p), 20) == picked, "not deterministic at one seed"


def test_reservoir_handles_a_file_smaller_than_the_sample(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"text":"only"}\n', encoding="utf-8")
    assert ev._reservoir(str(p), 20) == ['{"text":"only"}']


def test_load_or_split_streams_and_loses_nothing(tmp_path):
    p = tmp_path / "corpus.jsonl"
    lines = [json.dumps({"text": f"row {i}"}) for i in range(100)]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    train_path, held_path = ev._load_or_split(str(p), 0.1)
    train = list(ev._iter_jsonl(train_path))
    held = list(ev._iter_jsonl(held_path))

    assert len(held) == 10
    assert len(train) == 90
    assert set(train) | set(held) == set(lines), "a row went missing in the split"
    assert not (set(train) & set(held)), "a row is in both halves"


def test_load_or_split_on_an_empty_corpus(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    train_path, held_path = ev._load_or_split(str(p), 0.1)
    assert train_path == str(p)


def test_digest_is_short_and_stable():
    d = ev._digest("some corpus row that is quite long " * 20)
    assert len(d) == 40
    assert d == ev._digest("some corpus row that is quite long " * 20)


# ── an OOM is a measurement outcome ───────────────────────────────────────

def test_is_oom_recognises_torch_by_name_and_by_message():
    class OutOfMemoryError(Exception):
        pass
    assert ev._is_oom(OutOfMemoryError("CUDA out of memory"))
    assert ev._is_oom(MemoryError())
    assert ev._is_oom(RuntimeError(
        "CUDA out of memory. Tried to allocate 17974689792 bytes"))
    assert not ev._is_oom(ValueError("no router_logits"))
    assert not ev._is_oom(RuntimeError("shape mismatch"))


class _StubTok:
    eos_token_id = 0
    eos_token = ""

    def __call__(self, text, **kw):
        return {"input_ids": _StubTensor()}

    def decode(self, *a, **kw):
        return "whatever"


class _StubTensor:
    shape = (1, 4)

    def to(self, *a, **kw):
        return self

    def __getitem__(self, item):
        return self


class _OomModel:
    device = "cpu"

    def generate(self, **kw):
        raise RuntimeError(
            "CUDA out of memory. Tried to allocate 17974689792 bytes "
            "(free: 7906271232, total: 130663165952)")

    def eval(self):
        return self


@pytest.fixture
def fake_torch(monkeypatch):
    mod = types.ModuleType("torch")
    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    mod.no_grad = _NoGrad
    mod.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", mod)
    tf = types.ModuleType("transformers")
    tf.AutoModelForCausalLM = object
    tf.AutoTokenizer = object
    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.setattr(ev, "_torch_available", lambda: (True, ""))
    return mod


def test_oom_during_generation_is_unmeasurable_not_a_crash(tmp_path, fake_torch):
    corpus = tmp_path / "held.jsonl"
    corpus.write_text(
        json.dumps({"prompt": "def f():", "answer": "    return 1"}) + "\n",
        encoding="utf-8")
    model_dir = tmp_path / "expert"
    model_dir.mkdir()

    res = ev.eval_generation(
        model_dir=str(model_dir), test_data_path=str(corpus),
        label="python", domain="python", num_samples=1,
        loaded=(_OomModel(), _StubTok(), "cpu"))

    assert res.status == ev.UNMEASURABLE
    assert "out of memory" in res.note.lower()
    assert "17974689792" in res.note, "the note must carry the real allocation"
    assert res.exact_match in (None, 0.0) or res.exact_match == 0.0


# ── the prompt cap must be enforced where it cannot be argued with ────────

class _LongTensor:
    """A tensor-ish whose length is a lie the tokenizer told."""

    def __init__(self, n):
        self.shape = (1, n)

    def to(self, *a, **kw):
        return self

    def __getitem__(self, item):
        # batch[:, :1024] - report the sliced length
        if isinstance(item, tuple) and isinstance(item[1], slice):
            return _LongTensor(item[1].stop)
        return self


class _UntruncatingTok(_StubTok):
    """Ignores max_length, exactly as the 18 GB allocation implies."""

    def __call__(self, text, **kw):
        return {"input_ids": _LongTensor(25336),
                "attention_mask": _LongTensor(25336)}


class _RecordingModel:
    device = "cpu"

    def __init__(self):
        self.seen = None

    def generate(self, **kw):
        self.seen = kw["input_ids"].shape[-1]
        return _StubTensor()

    def eval(self):
        return self


def test_a_lying_tokenizer_cannot_hand_the_model_25000_tokens(tmp_path,
                                                              fake_torch):
    corpus = tmp_path / "held.jsonl"
    corpus.write_text(
        json.dumps({"prompt": "def f():", "answer": "    return 1"}) + "\n",
        encoding="utf-8")
    model_dir = tmp_path / "moe"
    model_dir.mkdir()
    model = _RecordingModel()

    ev.eval_generation(
        model_dir=str(model_dir), test_data_path=str(corpus),
        label="moe", domain="python", num_samples=1,
        loaded=(model, _UntruncatingTok(), "cpu"))

    assert model.seen == 1024, (
        f"model was handed {model.seen} tokens with max_prompt_tokens=1024 - "
        f"the second cap is the one that has to hold")


# ── the balloon must be visible and must be deflatable ────────────────────

def test_entry_point_sets_expandable_segments_before_torch(monkeypatch):
    """The allocator policy is useless if it lands after CUDA has initialised.

    It has to be in the environment at import time of the entry point, which
    is the only place guaranteed to run before anything imports torch.
    """
    import importlib
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    import ms_moe_maker.__main__ as m
    importlib.reload(m)
    import os
    assert os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True"


def test_an_existing_alloc_conf_is_never_overridden(monkeypatch):
    """A default, not a mandate. Somebody tuning their own allocator wins."""
    import importlib
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    import ms_moe_maker.__main__ as m
    importlib.reload(m)
    import os
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128"


def test_deflate_does_nothing_when_the_allocator_is_healthy(monkeypatch):
    # reserved barely above allocated: reusing its own blocks, leave it alone.
    monkeypatch.setattr(ev, "_balloon", lambda: (8192.0, 6000.0))
    called = []
    monkeypatch.setattr(ev, "_trace", lambda tag: called.append(tag))
    assert ev._deflate() is False
    assert not called


def test_deflate_skips_small_reservations(monkeypatch):
    # 300 MiB reserved for 1 MiB live is a 300x ratio and completely harmless.
    monkeypatch.setattr(ev, "_balloon", lambda: (300.0, 1.0))
    assert ev._deflate() is False


def test_sampler_attributes_a_peak_to_the_phase_it_happened_in():
    s = ev._MemSampler(interval=0.01)
    s.peaks = {}
    s.mark("load")
    s.peaks["load"] = {"cuda_alloc": 1000.0, "cuda_reserved": 1200.0,
                       "proc_rss": 500.0, "min_avail": 110000.0}
    s.mark("generate")
    s.peaks["generate"] = {"cuda_alloc": 6400.0, "cuda_reserved": 106600.0,
                           "proc_rss": 3000.0, "min_avail": 7000.0}
    table = s.table()
    assert "generate" in table and "106,600" in table
    assert "footprint" in table, "reserved+rss must be the reported total"
    # the balloon is the gap between these two columns; if the table ever
    # stops showing both, the diagnosis becomes invisible again
    assert "6,400" in table and "reserved" in table


# ── BLEU must be answerable for what it left out ──────────────────────────

def test_bleu_penalises_a_short_hedge():
    """Three right words out of a forty-word reference is not 1.00.

    Without a brevity penalty this returned pure precision, and on the first
    real 0.5B run that made BLEU the highest number on the board (0.877)
    sitting next to a ROUGE-1 of 0.369 saying most of the reference was never
    written. A metric that rewards saying less is worse than no metric.
    """
    reference = " ".join(f"word{i}" for i in range(40))
    hedge = "word0 word1 word2"
    assert ev._bleu_simple(hedge, reference) < 0.2

    full = reference
    assert ev._bleu_simple(full, reference) == pytest.approx(1.0)


def test_bleu_does_not_penalise_a_long_answer():
    reference = "alpha beta gamma"
    verbose = "alpha beta gamma delta epsilon zeta"
    # precision 0.5, no brevity penalty applies when c >= r
    assert ev._bleu_simple(verbose, reference) == pytest.approx(0.5)
