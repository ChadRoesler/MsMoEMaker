"""`ms-moe-maker build` - the full pipeline entry: plan, dryrun, or run."""
from __future__ import annotations

from pathlib import Path

from ..events import Events
from ._common import _load_recipe


def _cmd_build(args):
    """Run the full build pipeline.

    THIS GOES THROUGH Runner, and that is the whole fix. It used to call
    builder.run_pipeline() directly, which skipped the layer that emits the
    --json event stream, writes msmoe-run.json, and carries the refusal list.
    Runner was not dead code - it was the entrypoint layer, complete and
    orphaned, with a run_builder() branch already wired for the in-package
    path. Nothing needed building; something needed calling.

    Concretely, bypassing it cost three things at once:
      * --json events, which seren-theatre's stagehand consumes
      * the run manifest, which Theatre treats as authoritative when present
      * levers.Translation, i.e. every refusal the recipe earned
    """
    rec, errs, warns = _load_recipe(args.recipe, defaults_path=getattr(args, 'defaults', None))
    if rec is None:
        return 1

    events = Events(enabled=bool(args.json))

    # Prose goes to stderr whenever the machine stream owns stdout, so a
    # consumer can parse stdout without a heuristic for "is this line prose".
    say = events.say if args.json else print

    for w in warns:
        say(f"  · {w}")
        events.warning(w)

    from ..config import build_config
    from ..levers import translate
    from ..runner import Runner

    config = build_config(rec, force=args.force, dryrun=args.dryrun)
    translation = translate(rec, force=args.force, dryrun=args.dryrun)

    # TWO IDS, TWO QUESTIONS. recipe_id answers "is this the recipe you sent
    # me"; build_id answers "will my machine build what yours did". They stopped
    # being the same question when defaults moved onto the box.
    try:
        from ..config import build_id as _bid
        _build = _bid(config)
    except Exception:
        _build = "?"
    say(f"Ms.MoE — {config.name}  size={config.size}  tier={config.tier}")
    say(f"  ids      recipe {rec.recipe_id()}   build {_build}")
    # WHAT THAT TIER MEANS ON THIS BOX. The name alone stopped being enough the
    # moment a defaults file could redefine one: two machines can both say
    # `tier=spark` and mean different sizes, ranks and quants.
    try:
        from ..config import tier_table as _tt
        _spec = _tt(rec)[config.tier]
        say(f"  tier     {config.tier}: {_spec.max_vram_gb} GB, "
            f"default {_spec.default_size}, lora_r {_spec.default_lora_r}, "
            f"{_spec.default_quant}")
    except Exception:
        pass
    say(f"  base     {config.base}")
    say(f"  experts  {config.expert_names}")
    say(f"  steps    {config.target_steps}  "
        f"batch={config.per_device_batch}x{config.grad_accum}"
        f"  seq={config.max_seq_length}")
    # THE VOLUME, SAID OUT LOUD. "a real run but small" and "a full production
    # run" differ only in these numbers, and reading them back is the only way
    # to know which one you are about to start.
    say(f"  corpus   {config.min_samples_per_expert:,}-{config.num_code_samples:,}"
        f" samples/expert, {config.collect_token_target/1e6:.1f}M tokens target,"
        f" router mix {config.router_mix_total:,}")

    # THE GENERATED VOLUME, SAID OUT LOUD TOO — and it is the half that costs.
    #
    # The line above is documents COLLECTED. Every generated corpus is a
    # teacher running under rejection sampling, i.e. the most expensive text in
    # the build, and it was the only volume this header never mentioned. On the
    # first gauntlet-0.5B run the header read "2,976-12,000 samples/expert"
    # while the generator went for 15,000, and nothing on screen disagreed with
    # it. A header whose whole job is "which run am I about to start" must not
    # be silent about the priciest number in the answer.
    _gen = {}
    if config.tools_expert_name and config.tools_expert_name in config.expert_names:
        _gen[config.tools_expert_name] = "tools"
    for _e in getattr(rec, "experts", None) or []:
        if _e.name not in config.expert_names:
            continue
        if _e.name in (config.reasoning_experts or ()):
            _gen[_e.name] = "reasoning"
        elif getattr(getattr(_e, "source", None), "kind", "") == "synth":
            _gen.setdefault(_e.name, "synth")
    if _gen:
        say(f"  synth    {config.num_agent_samples:,} traces each for "
            + ", ".join(f"{n} ({k})" for n, k in _gen.items()))

    # WHICH DELIMITERS THIS RUN WILL SPLIT ON.
    #
    # A wrong tag style is a silent wrong ANSWER, not a crash: the splitter
    # finds no delimiters, reports "did not reason", and the whole think block
    # gets scored as if it were the answer. Every quality number in the run is
    # then wrong and the build looks fine. That is the one thing here nobody
    # can check afterwards, so it is the one most worth printing beforehand.
    if config.reasoning_type or config.reasoning_experts:
        _tags = (f"{config.reasoning_open}…{config.reasoning_close}"
                 if config.reasoning_open else "(no delimiters resolved)")
        _baked = (f", baked into {', '.join(config.reasoning_experts)}"
                  if config.reasoning_experts else "")
        say(f"  reasoning base={'yes' if config.reasoning else 'no'}, "
            f"style={config.reasoning_type or '(none)'}  {_tags}{_baked}")
    prov = getattr(rec, "defaults_provenance", None) or {}
    if prov:
        # LAYERED CONFIG WITHOUT PROVENANCE IS A SEANCE. A value that came from
        # a file the recipe never mentions has to say which file, or "why did
        # mine come out different" has no answer that is not archaeology.
        say("  defaults")
        # TERSE HERE, EXHAUSTIVE IN `validate`. --plan is the pre-flight read;
        # a five-field tier definition should not push the disk checks off the
        # screen. `validate` is the command you run when something is
        # surprising, so that one lists every leaf.
        from .. import defaults as _dm
        _blocks, _leaves = {}, []
        for _k, _v in sorted(prov.items()):
            _top = _k.split(".")[0]
            if _top in _dm.BOX_ONLY:
                _blocks.setdefault(".".join(_k.split(".")[:2]), [_v, 0])[1] += 1
            else:
                _leaves.append((_k, _v))
        for _k, _v in _leaves:
            say(f"    {_k:28} <- {_v}")
        for _k, (_v, _n) in sorted(_blocks.items()):
            say(f"    {_k + f' ({_n} fields)':28} <- {_v}")
        # The wire gets the FULL provenance even though the prose is terse:
        # a screen has a width, a consumer does not.
        events.emit("defaults", provenance=prov,
                    files=dict(getattr(rec, "defaults_digests", None) or {}))
    if config.floor_raised:
        say(f"  floor    corpus floor raised to "
            f"{config.min_samples_per_expert:,} docs/expert so the "
            f"{config.router_mix_total:,}-row router mix can be filled from "
            f"the .train split")
    say(f"  data     {config.data_root}")
    say(f"  output   {config.output_root}"
        + ("   [dryrun rung]" if config.dryrun else ""))

    if translation.refusals:
        # Named out loud, on both channels. A refusal the user cannot see is
        # the same as not having checked.
        events.refused(translation.refusals)
        say(f"  REFUSED ({len(translation.refusals)}):")
        for r in translation.refusals:
            say(f"    ✗ {r}")

    # --pipeline selects the LEGACY subprocess path (fork the old
    # fraunkenstein_universal.py) instead of the in-package builder. It used to
    # be validated here and then dropped on the floor - two sources disagreeing
    # inside one function - so a user who passed it got the in-package build
    # anyway, silently. Runner.run() dispatches on it: a real file means
    # run_subprocess, None means run_builder.
    pipeline = None
    if args.pipeline:
        pipeline = Path(args.pipeline)
        if not pipeline.is_file():
            raise SystemExit(f"--pipeline {pipeline} does not exist")

    if args.plan:
        # The laptop answer: what would this cost, what will it refuse, and
        # what would stop it. Preflight runs here too - the whole point is
        # that it costs nothing, so there is no reason to make someone start a
        # build to find out their corpus path is wrong.
        from .. import preflight as _pf
        from .. import stages as _st

        # NOT offline. --plan exists to answer "what would stop this", and a
        # dead model or dataset id is the most common answer. It used to skip
        # every reachability check, so a plan could come back clean and the
        # build then die at stage 1 on a repo that does not exist - which is
        # exactly what happened on the first real run.
        #
        # `validate` stays network-free: that is the laptop promise, and
        # corpus.py's Kind contract is declarative precisely so validation can
        # be answered by reading. --offline restores the old behaviour here.
        checks = _pf.run(config, rec, offline=args.offline, need_exporter=True)
        say("")
        for line in _pf.render(checks):
            say(line)
        synth = [e.name for e in rec.experts
                 if getattr(getattr(e, "source", None), "kind", "") == "synth"]
        say("")
        for i, (sid, label) in enumerate(
                _st.plan(config.expert_names, synth,
                         abliterate=config.abliterate_enabled), 1):
            say(f"  {i:>2}. {sid:<28} {label}")
        say(f"\n  {len(translation.agreed)} field(s) honoured, "
            f"{len(translation.refusals)} refused. Nothing was run.")
        events.done(ok=True, run_dir=config.output_root, stages_done=0,
                    stages_total=len(_st.plan(config.expert_names, synth,
                                              abliterate=config.abliterate_enabled)),
                    refusals=len(translation.refusals), planned_only=True)
        return 0

    if args.dryrun:
        say("[dryrun] smallest rung - this is a real build, just a cheap one")

    runner = Runner(
        rec, pipeline, translation, events,
        dryrun=args.dryrun,
        python=args.python,
    )

    # RESUMING INTO A DIFFERENT BUILD. Stages self-skip on artifacts found on
    # disk, so a changed knob plus a half-finished run produces a model whose
    # specialists were trained differently from each other - silently. Refuse
    # only when something is ALREADY DONE and would therefore be inherited
    # under the old settings; a fresh directory just gets restamped.
    changed, finished = runner.drift()
    if changed and finished and not translation.force:
        say("\n  REFUSING TO RESUME: this run directory was built by a "
            "different build.")
        say(f"  {len(finished)} stage(s) already finished and would be kept "
            f"as-is: {', '.join(finished)}")
        say("\n  What changed:")
        for c in changed:
            say(f"    · {c}")
        say("\n  Pick one:")
        say("    --force                 rebuild everything with the new settings")
        say("    --defaults <the old file>   reproduce the original build")
        say("    build somewhere else    change roots.output, keep both")
        events.error(stage="build",
                     message="run directory belongs to a different build_id")
        events.done(ok=False, errors=1, warnings=0)
        return 1
    if changed:
        say("  note: this run directory's settings changed since the last "
            "attempt, but nothing had finished yet - restamping.")
        for c in changed:
            say(f"    · {c}")

    return runner.run()
