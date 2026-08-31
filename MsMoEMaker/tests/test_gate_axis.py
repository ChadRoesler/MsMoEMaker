"""The gate axis belongs to the model, not the recipe."""
import json
import os

from ms_moe_maker.moe import stitch as st_mod
from ms_moe_maker.eval.harness import run_eval


class _Cfg:
    force = False

    def __init__(self, tmp, names, on_disk=None, moe_names=None):
        self.output_root = str(tmp / "run")
        self.data_root = str(tmp / "data")
        self.base = "b"
        self.expert_names = list(names)
        (tmp / "run").mkdir(parents=True, exist_ok=True)
        (tmp / "data").mkdir(parents=True, exist_ok=True)
        for n in names:
            (tmp / "data" / f"{n}.jsonl").write_text(
                json.dumps({"text": "a\nb\nc\nd\ne\nf"}) + "\n", encoding="utf-8")
        if on_disk is not None:
            d = tmp / "run" / "moe_untrained"
            d.mkdir(parents=True, exist_ok=True)
            (d / "config.json").write_text(
                json.dumps({"expert_names": on_disk}), encoding="utf-8")
            # A REAL SKELETON ALSO SAYS WHAT IT WAS SPLICED FROM. Without the
            # provenance stamp stitch_is_done declines (fail closed), which
            # would make every test in this file pass for the wrong reason.
            # Build the whole artifact, then ask the name question.
            for n in on_disk:
                sd = st_mod._specialist_dir(str(tmp / "run"), n)
                os.makedirs(sd, exist_ok=True)
                with open(os.path.join(sd, "config.json"), "w",
                          encoding="utf-8") as fh:
                    fh.write("{}")
            st_mod.write_provenance(str(d), str(tmp / "run"), list(on_disk))
        if moe_names is not None:
            d = tmp / "run" / "moe_trained"
            d.mkdir(parents=True, exist_ok=True)
            (d / "config.json").write_text(
                json.dumps({"expert_names": moe_names}), encoding="utf-8")


def test_a_reordered_recipe_does_not_reuse_the_old_skeleton(tmp_path):
    """The exact failure: expert list reordered, moe_trained deleted,
    moe_untrained left behind, stitch skipped, router trained on a gate axis
    that no longer matched the names."""
    cfg = _Cfg(tmp_path, ["csharp", "python"], on_disk=["python", "csharp"])
    assert st_mod.stitch_is_done(cfg) is False


def test_a_matching_skeleton_still_skips(tmp_path):
    cfg = _Cfg(tmp_path, ["python", "csharp"], on_disk=["python", "csharp"])
    assert st_mod.stitch_is_done(cfg) is True


def test_a_skeleton_without_names_is_left_alone(tmp_path):
    cfg = _Cfg(tmp_path, ["python", "csharp"], on_disk=[])
    assert st_mod.stitch_is_done(cfg) is True


def test_eval_believes_the_model_over_the_recipe(tmp_path):
    cfg = _Cfg(tmp_path, ["csharp", "python"], moe_names=["python", "csharp"])
    rep = run_eval(cfg, {"mode": "routing"})
    assert any("expert ORDER on disk" in c for c in rep.caveats), rep.caveats


def test_eval_refuses_to_label_a_different_model(tmp_path):
    cfg = _Cfg(tmp_path, ["csharp", "python"], moe_names=["rust", "go"])
    rep = run_eval(cfg, {"mode": "routing"})
    assert rep.routing.get("status") == "unmeasurable"
    assert "different models" in rep.routing.get("reason", "")
    assert any("REFUSED to label routing" in c for c in rep.caveats)


def test_agreement_is_silent(tmp_path):
    cfg = _Cfg(tmp_path, ["python", "csharp"], moe_names=["python", "csharp"])
    rep = run_eval(cfg, {"mode": "routing"})
    assert not any("ORDER on disk" in c for c in rep.caveats), rep.caveats
