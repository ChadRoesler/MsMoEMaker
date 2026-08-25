"""The `.env` loader, and the precedence rule it exists to keep simple.

huggingface_hub/datasets read HF_TOKEN natively; this just lets a box carry the
token (and other knobs) in a file instead of a shell export. Pure stdlib, no
torch, no network — so `validate` keeps its laptop promise.
"""

from ms_moe_maker.dotenv import _parse, load_dotenv


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


def test_load_dotenv_missing_file_is_empty(monkeypatch):
    # A path that cannot exist -> nothing applied.
    assert load_dotenv("__no_such_dir__/.env") == {}


def test_load_dotenv_shell_wins(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=from_file\nHF_HOME=/from/file\n", encoding="utf-8")

    monkeypatch.setenv("HF_TOKEN", "from_shell")
    applied = load_dotenv(str(env_file))

    # the shell's HF_TOKEN is left alone; the unset one comes from the file
    assert applied == {"HF_HOME": "/from/file"}
    assert monkeypatch.getenv("HF_TOKEN") == "from_shell"
    assert monkeypatch.getenv("HF_HOME") == "/from/file"
