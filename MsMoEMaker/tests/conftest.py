"""Fixtures shared by the tests that walk a synth loop without a GPU.

THE PROPERTY BEING PROTECTED IS THAT THIS SUITE NEEDS NO MODEL WHEEL.
`test_no_leaked_imports` asserts that importing this package does not drag
transformers in, and the reason it matters is `ms-moe-maker validate` running
on a laptop with no CUDA. A test that walks one of the generator loops still
passes through an AutoTokenizer call, because every loop loads the BASE
tokenizer before it builds a teacher - the corpus has to speak the base's
special tokens, not the teacher's. Stubbing it keeps those tests runnable in
the same bare interpreter as everything else.
"""

import sys
import types

import pytest


@pytest.fixture
def no_transformers(monkeypatch):
    """A stand-in `transformers`, so a loop test needs no model and no wheel."""
    fake = types.ModuleType("transformers")

    class _Tok:
        eos_token_id = 0
        eos_token = "<|im_end|>"
        pad_token_id = 0
        model_max_length = 4096
        # The agent loop REFUSES a base with no chat_template, loudly, because
        # it used to catch its own RuntimeError and carry on "because the
        # template check happens later". It never happened later.
        chat_template = "{{ messages }}"

        def apply_chat_template(self, msgs, **kw):
            """Render the messages, so a test can assert on what was WRITTEN.

            Returning a constant made every corpus row identical, which meant a
            test could confirm a row existed and nothing about its contents -
            and the contents are the whole question for an assembled row.
            """
            if isinstance(msgs, (list, tuple)):
                return "".join(
                    "<|%s|>%s" % (m.get("role", "?"), m.get("content", ""))
                    for m in msgs if isinstance(m, dict))
            return "PROMPT"

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(*a, **kw):
            return _Tok()

    fake.AutoTokenizer = _AutoTokenizer
    fake.set_seed = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "transformers", fake)
    return fake
