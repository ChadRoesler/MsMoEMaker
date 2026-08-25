"""The `.env` loader, and the precedence rule it exists to keep simple.

huggingface_hub/datasets read HF_TOKEN natively; this just lets a box carry the
token (and other knobs) in a file instead of a shell export. Pure stdlib, no
torch, no network — so `validate` keeps its laptop promise.

The file-reading tests use a checked-in fixture (`fixtures/sample.env`) rather
than `tmp_path`, so they run anywhere and actually exercise the read path.
"""
import os
from pathlib import Path

from ms_moe_maker.dotenv import _parse, load_dotenv

FIXTURE = Path(__file__).parent / "fixtures" / "sample.env"


def test_parse_basic():
    assert _parse("HF_TOKEN=hf_abc123") == ("HF_TOKEN", "hf_abc123")


def test_parse_strips_matching_quotes():
    assert _parse('HF_TOKEN="hf_abc123"') == ("HF_TOKEN", "hf_abc123")
    assert _parse("HF_TOKEN='hf_abc123'") == ("HF_TOKEN", "hf_abc123")


def test_parse_accepts_export_prefix():
    assert _parse("export HF_TOKEN=hf_abc123") == ("HF_TOKEN", "hf_abc123")


def test_parse_skips_comments_blanks_and_malformed():
    assert _parse("# a comment") is None
    assert _parse("") is None
    assert _parse("    ") is None
    assert _parse("HF_TOKEN") is None          # no '='
    assert _parse("=value") is None            # empty key


def test_load_dotenv_missing_file_is_empty():
    assert load_dotenv("__no_such_dir__/.env") == {}


def test_load_dotenv_reads_a_file(monkeypatch):
    monkeypatch.delenv("MSMOE_DOTENV_TEST", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    applied = load_dotenv(str(FIXTURE))
    assert applied == {"MSMOE_DOTENV_TEST": "from_file", "HF_TOKEN": "from_file"}


def test_load_dotenv_shell_wins(monkeypatch):
    monkeypatch.delenv("MSMOE_DOTENV_TEST", raising=False)
    monkeypatch.setenv("HF_TOKEN", "from_shell")
    applied = load_dotenv(str(FIXTURE))

    # HF_TOKEN is already in the environment, so it is skipped; the rest load.
    assert applied == {"MSMOE_DOTENV_TEST": "from_file"}
    assert os.environ["HF_TOKEN"] == "from_shell"
