"""Resume must recognise a SHARDED checkpoint as a finished one.

`save_pretrained` writes a single `model.safetensors` up to max_shard_size
(5GB by default) and, above that, `model-0000N-of-0000M.safetensors` shards
plus `model.safetensors.index.json`. A 7B in bf16 is ~15GB. So every size the
defaults table lists past 0.5B is saved sharded - and every resume predicate
in the pipeline looked for the single file only.

On a 0.5B that is the file that exists, so nothing noticed. On a 7B all three
predicates were permanently False: every resume retrained every specialist,
retrained the router and re-ran the whole abliteration study. The abliterate
predicate's own comment describes that exact failure as already fixed once,
which is why there is now ONE helper (stages.weights_present) and these tests
drive each caller through it rather than trusting that the fix reached all
three.
"""
from __future__ import annotations

import types

import pytest

from ms_moe_maker.abliterate import stage as abl
from ms_moe_maker.run import stages as st
from ms_moe_maker.train import finetune as ft
from ms_moe_maker.train import router as rt


def _sharded(d):
    """The layout transformers writes past max_shard_size: shards, then index."""
    (d / "model-00001-of-00002.safetensors").write_bytes(b"x")
    (d / "model-00002-of-00002.safetensors").write_bytes(b"x")
    (d / "model.safetensors.index.json").write_text(
        '{"weight_map": {}}', encoding="utf-8")


class TestTheHelper:
    def test_single_file_is_present(self, tmp_path):
        (tmp_path / "model.safetensors").write_bytes(b"x")
        assert st.weights_present(str(tmp_path))

    def test_sharded_is_present(self, tmp_path):
        _sharded(tmp_path)
        assert st.weights_present(str(tmp_path))

    def test_legacy_bin_both_layouts(self, tmp_path):
        (tmp_path / "pytorch_model.bin").write_bytes(b"x")
        assert st.weights_present(str(tmp_path))
        (tmp_path / "pytorch_model.bin").unlink()
        (tmp_path / "pytorch_model.bin.index.json").write_text(
            "{}", encoding="utf-8")
        assert st.weights_present(str(tmp_path))

    def test_shards_without_their_index_are_not_a_finished_save(self, tmp_path):
        """transformers writes the index LAST, so shards with no index is the
        mid-save state - the exact window the presence-only bug lived in."""
        (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"x")
        assert not st.weights_present(str(tmp_path))

    def test_empty_and_missing_directories_are_absent(self, tmp_path):
        assert not st.weights_present(str(tmp_path))
        assert not st.weights_present(str(tmp_path / "nope"))

    def test_the_predicates_go_through_the_helper(self):
        """The wiring half: a fourth hand-written copy of the marker list is
        the way this regresses. Each caller has to name the helper."""
        import inspect
        for mod, fn in ((abl, abl.abliterate_is_done),
                        (ft, ft.specialist_is_done),
                        (rt, rt.router_is_done)):
            src = inspect.getsource(fn)
            assert "weights_present" in src, mod.__name__
            assert '"model.safetensors"' not in src, (
                f"{mod.__name__} still checks the single-file layout by hand")


class TestAbliterate:
    def test_a_sharded_base_counts_as_done(self, tmp_path):
        cfg = types.SimpleNamespace(force=False, output_root=str(tmp_path))
        d = tmp_path / abl.abliterate_dir(cfg)
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}", encoding="utf-8")
        (d / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        assert abl.abliterate_is_done(cfg) is False, "no weights yet"
        _sharded(d)
        assert abl.abliterate_is_done(cfg) is True, (
            "a sharded abliterated base re-ran the whole study on every resume")


class TestSpecialist:
    def test_a_sharded_specialist_counts_as_trained(self, tmp_path):
        cfg = types.SimpleNamespace(force=False, output_root=str(tmp_path))
        d = tmp_path / ft.specialist_dir(cfg, "python")
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}", encoding="utf-8")
        (d / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        assert ft.specialist_is_done(cfg, "python") is False
        _sharded(d)
        assert ft.specialist_is_done(cfg, "python") is True

    def test_retrain_still_wins_over_a_sharded_specialist(self, tmp_path):
        """--only <expert> must keep forcing, whatever layout is on disk."""
        cfg = types.SimpleNamespace(force=False, output_root=str(tmp_path))
        d = tmp_path / ft.specialist_dir(cfg, "python")
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}", encoding="utf-8")
        (d / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        _sharded(d)
        assert ft.specialist_is_done(cfg, "python", retrain=True) is False


class TestRouter:
    def test_a_sharded_router_counts_as_trained(self, tmp_path):
        """The trained MoE is N specialists wide, so it shards long before the
        specialists do - this predicate was the first to go wrong."""
        cfg = types.SimpleNamespace(force=False, output_root=str(tmp_path))
        d = tmp_path / rt.router_dir(cfg)
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}", encoding="utf-8")
        assert rt.router_is_done(cfg) is False
        _sharded(d)
        assert rt.router_is_done(cfg) is True

    def test_force_still_wins(self, tmp_path):
        cfg = types.SimpleNamespace(force=True, output_root=str(tmp_path))
        d = tmp_path / rt.router_dir(cfg)
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}", encoding="utf-8")
        _sharded(d)
        assert rt.router_is_done(cfg) is False
