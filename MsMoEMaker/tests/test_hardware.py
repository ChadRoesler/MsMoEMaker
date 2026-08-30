"""Tests for the hardware tier definitions."""
import pytest

from ms_moe_maker.box import hardware


class TestTierSpec:
    """TierSpec dataclass and TIERS dict."""

    def test_nano_tier(self):
        spec = hardware.get_tier("nano")
        assert spec.max_vram_gb == 3
        assert spec.default_size == "3B"
        assert spec.default_lora_r == 32
        assert spec.default_quant == "Q4_K_M"
        assert "3B" in spec.recommended_sizes

    def test_xavier_tier(self):
        spec = hardware.get_tier("xavier")
        assert spec.max_vram_gb == 9
        assert spec.default_size == "7B"
        assert spec.default_lora_r == 64
        assert spec.default_quant == "Q5_K_M"
        assert "7B" in spec.recommended_sizes

    def test_spark_tier(self):
        spec = hardware.get_tier("spark")
        assert spec.max_vram_gb == 36
        assert spec.default_size == "32B"
        assert spec.default_lora_r == 128
        assert spec.default_quant == "Q8_0"
        assert spec.supports_fp8 is True
        assert "32B" in spec.recommended_sizes

    def test_tiers_dict(self):
        assert set(hardware.TIERS.keys()) == {"nano", "xavier", "spark"}

    def test_resolve_tier_explicit(self):
        assert hardware.resolve_tier("nano") == "nano"
        assert hardware.resolve_tier("xavier") == "xavier"
        assert hardware.resolve_tier("spark") == "spark"

    def test_resolve_tier_auto_vram(self):
        assert hardware.resolve_tier(detected_vram_gb=2) == "nano"
        assert hardware.resolve_tier(detected_vram_gb=3) == "nano"
        assert hardware.resolve_tier(detected_vram_gb=5) == "xavier"
        assert hardware.resolve_tier(detected_vram_gb=9) == "xavier"
        assert hardware.resolve_tier(detected_vram_gb=18) == "spark"
        assert hardware.resolve_tier(detected_vram_gb=50) == "spark"

    def test_resolve_tier_default(self):
        # No recipe tier, no VRAM → defaults to middle (xavier)
        assert hardware.resolve_tier() == "xavier"

    def test_tier_for_size(self):
        assert hardware.tier_for_size("0.5B") == "nano"
        assert hardware.tier_for_size("1.5B") == "nano"
        assert hardware.tier_for_size("3B") == "nano"
        assert hardware.tier_for_size("7B") == "xavier"
        assert hardware.tier_for_size("14B") == "spark"
        assert hardware.tier_for_size("32B") == "spark"

    def test_get_tier_unknown(self):
        with pytest.raises(ValueError, match="unknown tier"):
            hardware.get_tier("tiny")


class TestVramUniqueness:
    """VRAM values are distinct and ordered."""

    def test_all_tiers_have_unique_vram(self):
        vrams = [hardware.get_tier(n).max_vram_gb for n in ("nano", "xavier", "spark")]
        assert len(vrams) == len(set(vrams))

    def test_vram_order_is_increasing(self):
        tiers = sorted(
            ("nano", "xavier", "spark"),
            key=lambda n: hardware.get_tier(n).max_vram_gb,
        )
        assert tiers == ["nano", "xavier", "spark"]


class TestTiersStayReal:
    """The drift this table had: xavier/spark defaulted to 9B/36B, neither of
    which is a MODEL_SIZES key, so config would have built a base model that
    does not exist. Pin that every size a tier names actually resolves."""

    def test_every_default_and_recommended_size_is_real(self):
        from ms_moe_maker.config import pipeline as config
        for name in hardware.TIERS:
            spec = hardware.get_tier(name)
            assert spec.default_size in config.MODEL_SIZES, (
                f"{name}.default_size {spec.default_size!r} is not a MODEL_SIZES key")
            for size in spec.recommended_sizes:
                assert size in config.MODEL_SIZES, (
                    f"{name} recommends {size!r}, which is not a MODEL_SIZES key")

    def test_config_reads_hardware_not_a_copy(self, monkeypatch):
        """config.py used to keep _TIER_HINTS/_TIER_RANK and they drifted.
        Assert build_config resolves tier defaults from hardware.TIERS."""
        from ms_moe_maker.config import pipeline as config
        from ms_moe_maker.config.recipe import parse

        monkeypatch.delenv("MSMOE_DRYRUN", raising=False)
        monkeypatch.delenv("MSMOE_TIER", raising=False)
        monkeypatch.delenv("MSMOE_LORA_R", raising=False)

        body = {"schema_version": 1, "name": "t", "size": "auto",
                "experts": [{"name": "a", "source": {"kind": "hf", "repo": "o/d"}},
                            {"name": "b", "source": {"kind": "hf", "repo": "o/e"}}]}
        for tier in hardware.TIERS:
            body["runtime"] = {"hardware_tier": tier}
            rec, _ = parse(body)
            c = config.build_config(rec, dryrun=False)
            spec = hardware.get_tier(tier)
            assert c.tier == tier
            assert c.size == spec.default_size
            assert c.lora_r == spec.default_lora_r
            # 4-bit TRAINING is opt-in, never derived from the tier's GGUF
            # quant - deriving it made the nano floor unfinishable by default.
            assert c.load_in_4bit is False
