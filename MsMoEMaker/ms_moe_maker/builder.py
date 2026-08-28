"""Pipeline builder — orchestrates all stages end-to-end.

Reads a Recipe (recipe.py dataclass), resolves config (config.py), and
executes stages in order:

    preflight → collect corpora → generate agent traces
              → finetune each specialist
              → stitch MoE → train router → export GGUF

Uses callbacks to report progress back to the runner/manifest.  Each stage
skips if its artifact already exists (unless force=True).

This module is the PRODUCT.  The old fraunkenstein_universal.py was the
pipeline; this is the factory.  The pipeline modules (data.py, finetune.py,
stitch.py, router.py, export.py) are the machinery.
"""
from __future__ import annotations

import os
import sys
import time
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import config as cfg_module
from . import manifest as mf
from . import stages


class BuildResult:
    """Outcome of a build run."""

    def __init__(self):
        self.ok = False
        self.stages_completed: List[str] = []
        self.failed_stage: Optional[str] = None
        self.message: str = ""
        self.artifacts: Dict[str, str] = {}  # stage_id → path


class StageCallback:
    """Interface for reporting stage progress.

    The runner passes a callback that updates the manifest.  Each pipeline
    function receives it and calls stage(name, status, note) after doing work.
    """

    def __init__(self, notify=None):
        """
        notify: optional callable(stage_id, status, note) for propagating
                stage changes outward (to the Runner / manifest).
        """
        self._stages: Dict[str, dict] = {}
        self.notify = notify  # callable(stage_id, status, note) or None

    def stage(self, name: str, status: str, note: str = ""):
        if name not in self._stages:
            self._stages[name] = {"name": name, "status": "pending", "note": ""}
        self._stages[name]["status"] = status
        if note:
            self._stages[name]["note"] = note
        if self.notify:
            self.notify(name, status, note)


def run_pipeline(recipe, force: bool = False, dryrun: bool = False,
                 callback: Optional[StageCallback] = None) -> BuildResult:
    """Execute the full Ms.MoE pipeline.

    Args:
        recipe: A recipe.py Recipe dataclass.
        force: Redo stages whose artifacts exist.
        dryrun: Run on the smallest rung for structural testing.
        callback: StageCallback for progress reporting.

    Returns:
        BuildResult with outcomes.
    """
    from . import config as cfg_module
    from . import data as data_mod
    from . import finetune as finetune_mod
    from . import stitch as stitch_mod
    from . import router as router_mod
    from . import export as export_mod
    import torch

    result = BuildResult()
    cb = callback or StageCallback()
    t_start = time.time()

    # ── Resolve config ────────────────────────────────────────────────────
    # Build config from recipe + env overrides. llama.cpp is resolved here too
    # (recipe → env → search); there is no post-hoc override, because
    # PipelineConfig is frozen and an assignment to it raised FrozenInstanceError.
    config = cfg_module.build_config(recipe, force=force, dryrun=dryrun)

    # Create output directories
    os.makedirs(config.data_root, exist_ok=True)
    os.makedirs(config.output_root, exist_ok=True)
    os.makedirs(config.shard_cache, exist_ok=True)

    # Set HF_HOME if not already set
    hf_home = os.environ.get("HF_HOME", "")
    if not hf_home:
        os.environ["HF_HOME"] = config.hf_home

    # ── Preflight ─────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"Ms.MoE pipeline — size={config.size}  base={config.base}")
    print("=" * 60)

    # Print config stamp (the [cfg] block from the old script)
    print(f"[cfg] batch={config.per_device_batch}x{config.grad_accum} "
          f"(eff {config.per_device_batch * config.grad_accum})  "
          f"lora_r={config.lora_r} dropout={config.lora_dropout}  "
          f"4bit={config.load_in_4bit}")
    print(f"[cfg] rung: size={config.size}  target_steps={config.target_steps}  "
          f"token_budget/expert={config.expert_token_budget/1e6:.2f}M")
    print(f"[cfg] roots: data={config.data_root}")
    print(f"[cfg]        model={config.output_root}")

    # Report disk space
    for label, path in [("output  ", config.output_root),
                        ("shards  ", config.shard_cache),
                        ("hf cache", config.hf_home)]:
        try:
            usage = shutil.disk_usage(path)
            print(f"   {label} {path}  ({usage.free / 2**30:.0f} GB free)")
        except OSError as e:
            print(f"   {label} {path}  (unreadable: {e})")

    cb.stage(stages.PREFLIGHT, "running", "checking the box")

    # ASK EVERY CHEAP QUESTION NOW. This stage used to print the config stamp
    # and report done without checking anything, so a missing llama.cpp was
    # discovered at stage 6 - after every specialist had trained, the stitch
    # had landed and the router had run. Nothing before it was wasted, which
    # is exactly what made it infuriating: the answer was knowable before a
    # single token was read.
    from . import preflight as preflight_mod

    pf = preflight_mod.run(config, recipe)
    for line in preflight_mod.render(pf):
        print(line)

    if not pf.ok:
        summary = "; ".join(f"{c.name}: {c.detail}" for c in pf.failures)
        result.failed_stage = stages.PREFLIGHT
        result.message = f"preflight failed - {summary}"
        cb.stage(stages.PREFLIGHT, "failed", summary)
        return result

    note = "checks passed"
    if pf.warnings:
        note = (f"checks passed, {len(pf.warnings)} warning(s): "
                + "; ".join(c.name for c in pf.warnings))
    cb.stage(stages.PREFLIGHT, mf.DONE, note)
    result.stages_completed.append(stages.PREFLIGHT)

    # ── Stage: abliterate the base (optional) ───────────────────────────────
    #
    # Runs the vendored Heretic core on the resolved base and repoints
    # `config.base` at the result, so every specialist trains from the
    # decensored checkpoint. Skipped unless the recipe asks for it.
    if config.abliterate_enabled:
        import dataclasses

        from . import abliterate as abliterate_mod

        cb.stage(stages.ABLITERATE_BASE, "running", "decensoring the base model")
        ablated_dir = abliterate_mod.abliterate_base(config)
        config = dataclasses.replace(config, base=ablated_dir)
        print(f"[abliterate] base repointed -> {ablated_dir}")
        cb.stage(stages.ABLITERATE_BASE, mf.DONE, f"abliterated base → {ablated_dir}")
        result.stages_completed.append(stages.ABLITERATE_BASE)
        result.artifacts[stages.ABLITERATE_BASE] = ablated_dir

    # ── Determine expert list and sources ─────────────────────────────────
    expert_names = config.expert_names
    if not expert_names:
        expert_names = [e.name for e in recipe.experts]

    from .config import DISPLAY_LANG

    # ── Stage 1: Collect code corpora ─────────────────────────────────────
    cb.stage(stages.DATA_CORPUS, "running", "collecting expert corpora")
    print(f"\n{'=' * 60}")
    print(f"Stage 1: Collecting corpora for {len(expert_names)} experts")
    print(f"{'=' * 60}")

    # Build sources dict from recipe experts for data collection
    _sources: Dict[str, Any] = {}
    for e in recipe.experts:
        if e.source:
            _sources[e.name] = e.source

    code_paths = data_mod.collect_corpus(config, languages=expert_names,
                                         sources=_sources, callback=cb.stage)

    if not code_paths:
        result.failed_stage = stages.DATA_CORPUS
        result.message = "No corpora were collected"
        cb.stage(stages.DATA_CORPUS, "failed", result.message)
        return result

    cb.stage(stages.DATA_CORPUS, mf.DONE,
             f"collected {len(code_paths)} corpora")
    result.stages_completed.append(stages.DATA_CORPUS)
    result.artifacts[stages.DATA_CORPUS] = ", ".join(code_paths.values())

    # ── Stage 2: Generate agent traces ────────────────────────────────────
    has_synth = any(
        hasattr(e, "source") and hasattr(e.source, "kind")
        and e.source.kind == "synth"
        for e in recipe.experts
        if e.name in expert_names
    )

    agent_path = None
    synth_paths: Dict[str, str] = {}
    reasoning_paths: Dict[str, str] = {}
    if has_synth or config.tools_expert_name in expert_names or config.reasoning_experts:
        cb.stage(stages.DATA_SYNTH, "running", "generating synthetic corpora")
        print(f"\n{'=' * 60}")
        print(f"Stage 2: Generating synthetic corpora")
        print(f"{'=' * 60}")

        # EVERY GENERATED EXPERT, NOT JUST THE TOOLS ONE.
        #
        # This called generate_agent_traces exactly ONCE, always with
        # expert_name=tools_name, while `has_synth` above was computed across
        # ALL experts. So an expert with `source.kind: synth` under any other
        # name lit the stage up and had nothing generated for it: data.py's
        # collector skips kind=synth on purpose ("handled by
        # generate_agent_traces, not corpus"), so code_paths had no entry
        # either, and the build died hours later in the fine-tune loop at
        # "No data path for expert X" - after preflight, after abliterate.base,
        # after the entire corpus stage. Loud, but late, and `build --plan`
        # had reported `[ok] source/<name> kind=synth` on the way in.
        #
        # THIS IS THE SECOND HALF OF THE data.code -> data.corpus FIX. That one
        # taught stages.plan() that a generated expert can be called anything.
        # The worker behind the stage still assumed it was called "agentcore".
        # When you kill a hardcoded-name assumption, grep for every other place
        # that name is a default - the one you noticed is rarely the only one.
        #
        # A synth expert carrying `reasoning: true` is deliberately EXCLUDED
        # here: generate_reasoning_traces below writes its corpus and
        # reasoning_paths wins in the fine-tune loop, so generating tool traces
        # for it would be teacher hours spent on a file nothing ever reads.
        # The tools expert (agentcore) keeps the MCP tool-call path. Every OTHER
        # synth expert that is not reasoning gets PLAIN DOMAIN TEXT - answers to
        # the templates questions, no think block, no tool calls.
        tools_name = config.tools_expert_name  # '' when there is no tools expert
        domain_names = [
            e.name for e in recipe.experts
            if e.name in expert_names
            and getattr(getattr(e, "source", None), "kind", "") == "synth"
            and e.name not in (config.reasoning_experts or ())
            and e.name != tools_name
        ]

        if tools_name and tools_name in expert_names:
            agent_path = data_mod.generate_agent_traces(
                config, callback=cb.stage,
                expert_name=tools_name,
                teacher_model=cfg_module.teacher_for(recipe, config, tools_name))

        for sname in domain_names:
            spath = data_mod.generate_domain_traces(
                config, callback=cb.stage,
                expert_name=sname,
                teacher_model=cfg_module.teacher_for(recipe, config, sname),
                templates_path=cfg_module.templates_for(recipe, sname))
            if spath:
                synth_paths[sname] = spath

        for rname in config.reasoning_experts:
            rpath = data_mod.generate_reasoning_traces(
                config, rname, callback=cb.stage,
                teacher_model=cfg_module.teacher_for(recipe, config, rname),
                templates_path=cfg_module.templates_for(recipe, rname))
            if rpath:
                reasoning_paths[rname] = rpath

        produced = (([agent_path] if agent_path else [])
                    + list(synth_paths.values())
                    + list(reasoning_paths.values()))
        if produced:
            cb.stage(stages.DATA_SYNTH, "done",
                     f"synthetic corpora → {', '.join(produced)}")
            result.stages_completed.append(stages.DATA_SYNTH)
            result.artifacts[stages.DATA_SYNTH] = ", ".join(produced)
        else:
            cb.stage(stages.DATA_SYNTH, mf.SKIPPED, "synthetic corpora already present")
            result.stages_completed.append(stages.DATA_SYNTH)

    # ── Stage 3: Fine-tune specialists ────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Stage 3: Fine-tuning specialists")
    print(f"{'=' * 60}")

    specialist_dirs: Dict[str, str] = {}
    # The corpus each expert actually trained on. Tracked rather than
    # reconstructed later, because for a synth expert it is the generated
    # trace file and not code_paths[name] - so re-deriving it downstream would
    # quietly train the router on a different mix than the experts saw.
    expert_corpus_paths: Dict[str, str] = {}
    for i, safe_name in enumerate(expert_names):
        print(f"\n  [{i+1}/{len(expert_names)}] {safe_name}")
        cb.stage(f"finetune.{safe_name}", "running",
                 f"fine-tuning specialist {safe_name}")

        # Find data path for this expert
        data_path = code_paths.get(safe_name)
        if safe_name == config.tools_expert_name and agent_path:
            data_path = agent_path
        # A non-tools synth expert. Ordered before the reasoning check on
        # purpose: an expert can be both, and the reasoning corpus is the one
        # its specialist must actually learn from.
        if safe_name in synth_paths:
            data_path = synth_paths[safe_name]
        if safe_name in reasoning_paths:
            data_path = reasoning_paths[safe_name]
        if not data_path:
            # Check if expert has a custom data source
            for e in recipe.experts:
                if e.name == safe_name:
                    if hasattr(e, "source") and hasattr(e.source, "kind"):
                        if e.source.kind == "local":
                            data_path = e.source.path
                            break
            if not data_path:
                result.failed_stage = f"finetune.{safe_name}"
                result.message = f"No data path for expert {safe_name}"
                cb.stage(f"finetune.{safe_name}", "failed", result.message)
                return result

        # ASK BEFORE, REPORT AFTER. The stage function self-skips when the
        # artifact is already there, and reporting "done" for a stage that did
        # nothing erases the difference between a run and a resume. The legacy
        # subprocess path kept them distinct - the first real Spark run
        # correctly showed data.code as SKIPPED, not done - and the in-package
        # builder quietly lost it, so a resumed six-hour build claimed every
        # stage had executed.
        #
        # manifest.py has had SKIPPED the whole time, with the comment
        # "already present on disk; the pipeline's _done() fired". The
        # vocabulary was never the missing part.
        was_present = finetune_mod.specialist_is_done(config, safe_name)
        expert_corpus_paths[safe_name] = data_path
        out_dir = finetune_mod.fine_tune_specialist(
            config, safe_name, data_path,
            expert_display=DISPLAY_LANG.get(safe_name, safe_name),
        )
        specialist_dirs[safe_name] = out_dir
        cb.stage(f"finetune.{safe_name}",
                 mf.SKIPPED if was_present else mf.DONE,
                 f"already trained → {out_dir}" if was_present
                 else f"saved → {out_dir}")
        result.stages_completed.append(f"finetune.{safe_name}")
        result.artifacts[f"finetune.{safe_name}"] = out_dir

        # Memory cleanup after each expert.
        #
        # This was `del torch` followed two lines later by `torch.cuda...`,
        # which is an UnboundLocalError - `torch` is a function-local name
        # here, so deleting it unbinds it for the rest of the call. It fired
        # at the END of expert #1, i.e. after the most expensive stage in the
        # build, so a real run died having already burned the hours.
        #
        # It would not have freed anything either: `del` drops a NAME, and the
        # module stays in sys.modules regardless. What actually holds VRAM is
        # the model and the trainer, and those are local to
        # fine_tune_specialist - they are already out of scope by the time we
        # get here. So the honest cleanup is a collect and an empty_cache, and
        # nothing else. Keep it that way; a `del` here is always wrong.
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Gate: do the experts differ, and can a router learn from them? ────
    #
    # BEFORE THE STITCH, ON PURPOSE. Everything downstream of here - stitch,
    # router train, GGUF export, smoke, eval - runs identically on two experts
    # that learned different things and on two that learned the same thing.
    # The first time that happened it was found after all of it, by hand, with
    # three separate probes. This is those probes, run at the one moment the
    # answer can still save the work.
    #
    # WARN, NEVER REFUSE. Someone may genuinely want a dense ensemble, and
    # mandate is not ethos: the job is to make sure they know what they are
    # getting, not to decide for them.
    gate_mode = getattr(getattr(recipe, "gates", None), "experts", "auto")
    if len(specialist_dirs) >= 2 and gate_mode != "skip":
        cb.stage(stages.GATE_EXPERTS, "running", "comparing the specialists")
        try:
            from . import experts as experts_mod
            from . import eval as eval_mod

            held: Dict[str, str] = {}
            if gate_mode == "auto":
                data_root = Path(config.data_root)
                for name in specialist_dirs:
                    for cand in (data_root / f"{name}.jsonl",
                                 data_root / f"{name}_code.jsonl"):
                        if cand.is_file():
                            _, hp = eval_mod._load_or_split(
                                str(cand), config.eval_held_out_fraction)
                            held[name] = hp
                            break

            gate = experts_mod.run_experts(
                config, dict(specialist_dirs), held_paths=held or None,
                spec={"num_samples": 12}, callback=cb.stage)
            print(experts_mod.format_report(gate))
            result.artifacts[stages.GATE_EXPERTS] = gate.status
            if gate.findings:
                cb.stage(stages.GATE_EXPERTS, "done",
                         f"{len(gate.findings)} finding(s) - see the report; "
                         f"building anyway")
            else:
                cb.stage(stages.GATE_EXPERTS, "done", "experts look routable")
        except Exception as exc:                      # never fail the build
            # A GATE THAT CRASHES MUST NOT TAKE THE BUILD WITH IT. It is
            # advisory by design, so its own failure is reported as "not
            # measured" - the one thing it must never do is look like a pass.
            print(f"[warn] expert gate could not run: {exc}", file=sys.stderr)
            cb.stage(stages.GATE_EXPERTS, "skipped", f"gate error: {exc}")
    elif len(specialist_dirs) >= 2:
        cb.stage(stages.GATE_EXPERTS, "skipped", "gates.experts: skip")

    # ── Stage 4: Stitch MoE ───────────────────────────────────────────────
    cb.stage(stages.STITCH, "running", "stitching MoE skeleton")
    print(f"\n{'=' * 60}")
    print(f"Stage 4: Stitching MoE")
    print(f"{'=' * 60}")

    stitch_was_present = stitch_mod.stitch_is_done(config)
    moe_dir = stitch_mod.stitch_moe(config, list(specialist_dirs.keys()))
    if not moe_dir:
        result.failed_stage = stages.STITCH
        result.message = "MoE stitching failed"
        cb.stage(stages.STITCH, "failed", result.message)
        return result

    # Verify stitch
    # Pass output_root explicitly so verification finds the specialists to
    # compare against, rather than guessing from the MoE dir's parent.
    if stitch_mod.verify_stitch(moe_dir, output_root=config.output_root,
                                gate_fill=config.shared_expert_gate_fill,
                                router_init=getattr(config, "router_init",
                                                    "zero")):
        cb.stage(stages.STITCH,
                 mf.SKIPPED if stitch_was_present else mf.DONE,
                 f"skeleton → {moe_dir}")
        result.stages_completed.append(stages.STITCH)
        result.artifacts[stages.STITCH] = moe_dir
    else:
        result.failed_stage = stages.STITCH
        result.message = "MoE verification failed"
        cb.stage(stages.STITCH, "failed", result.message)
        return result

    # ── Stage 5: Train router ─────────────────────────────────────────────
    cb.stage(stages.ROUTER, "running", "training router")
    print(f"\n{'=' * 60}")
    print(f"Stage 5: Training router")
    print(f"{'=' * 60}")

    router_was_present = router_mod.router_is_done(config)
    # CORPORA, not specialist dirs. Passing specialist_dirs here is what made
    # a ten-minute run die on `open()` at stage 5.
    router_dir = router_mod.train_router(
        config, moe_dir, list(specialist_dirs.keys()), expert_corpus_paths)
    if not router_dir:
        result.failed_stage = stages.ROUTER
        result.message = "Router training failed"
        cb.stage(stages.ROUTER, "failed", result.message)
        return result

    cb.stage(stages.ROUTER,
             mf.SKIPPED if router_was_present else mf.DONE,
             f"router-trained → {router_dir}")
    result.stages_completed.append(stages.ROUTER)
    result.artifacts[stages.ROUTER] = router_dir

    # ── Stage 6: Export GGUF ──────────────────────────────────────────────
    cb.stage(stages.EXPORT_GGUF, "running", "exporting GGUF")
    print(f"\n{'=' * 60}")
    print(f"Stage 6: Exporting GGUF")
    print(f"{'=' * 60}")

    export_was_present = export_mod.export_is_done(config)
    gguf_path = export_mod.export_gguf(config, router_dir)
    if gguf_path is None:
        cb.stage(stages.EXPORT_GGUF, "warning", "GGUF export skipped (no llama.cpp)")
        result.stages_completed.append(stages.EXPORT_GGUF)
        result.artifacts[stages.EXPORT_GGUF] = "skipped (no llama.cpp)"
    elif gguf_path:
        cb.stage(stages.EXPORT_GGUF,
                 mf.SKIPPED if export_was_present else mf.DONE,
                 f"GGUF → {gguf_path}")
        result.stages_completed.append(stages.EXPORT_GGUF)
        result.artifacts[stages.EXPORT_GGUF] = gguf_path
    else:
        result.failed_stage = stages.EXPORT_GGUF
        result.message = "GGUF export failed"
        cb.stage(stages.EXPORT_GGUF, "failed", result.message)
        return result

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    result.ok = True
    result.message = (f"Build complete in {elapsed:.0f}s. "
                      f"{len(result.stages_completed)} stages done. "
                      f"GGUF: {gguf_path or 'skipped'}")
    print(f"\n{'=' * 60}")
    print(f"✨ Ms.MoE build complete in {elapsed:.0f}s")
    print(f"{'=' * 60}")
    print(f"   Specialist dirs: {len(specialist_dirs)}")
    for name, path in specialist_dirs.items():
        print(f"     {name:12} → {path}")
    print(f"   MoE skeleton:    {moe_dir}")
    print(f"   Router-trained:  {router_dir}")
    print(f"   GGUF:            {gguf_path or 'skipped'}")
    # Next steps point at THIS tool's own verbs, not at the Lab scripts they
    # were carved out of. A stranger who pip-installed ms-moe-maker has no
    # verify_stitch_complete.py and no eval_fraunkenstein.py, and telling them
    # to run one is telling them they are using someone else's tool.
    print(f"\n   Next steps:")
    print(f"     ms-moe-maker smoke <recipe>     # does the GGUF generate?")
    print(f"     ms-moe-maker eval  <recipe>     # expert-only vs MoE, dead-expert flag")

    return result
