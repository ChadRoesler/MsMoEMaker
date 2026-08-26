"""The `.env` loader, and the precedence rule it exists to keep simple.

huggingface_hub/datasets read HF_TOKEN natively; this just lets a box carry the
token (and other knobs) in a file instead of a shell export. Pure stdlib, no
torch, no network — so `validate` keeps its laptop promise.

The file-reading tests write their own `.env` into `tmp_path`, so they are
self-contained — no checked-in fixture to go stale or fail to travel to another
box.
"""
import os

from ms_moe_maker.dotenv import _parse, load_dotenv

# A key no real environment will ever set, so "reads a file" and "shell wins"
# are deterministic on any box.
_TEST_KEY = "MSMOE_DOTENV_TEST"


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


def test_load_dotenv_reads_a_file(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"{_TEST_KEY}=from_file\nHF_TOKEN=from_file\n",
                        encoding="utf-8")
    monkeypatch.delenv(_TEST_KEY, raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    applied = load_dotenv(str(env_file))
    assert applied == {_TEST_KEY: "from_file", "HF_TOKEN": "from_file"}


def test_load_dotenv_shell_wins(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"{_TEST_KEY}=from_file\nHF_TOKEN=from_file\n",
                        encoding="utf-8")
    monkeypatch.delenv(_TEST_KEY, raising=False)
    monkeypatch.setenv("HF_TOKEN", "from_shell")
    applied = load_dotenv(str(env_file))

    # HF_TOKEN is already in the environment, so it is skipped; the rest load.
    assert applied == {_TEST_KEY: "from_file"}
    assert os.environ["HF_TOKEN"] == "from_shell"
