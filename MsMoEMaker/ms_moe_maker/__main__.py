"""Ms.MoE Maker — build targeted Mixtures of Experts from a recipe.

    ms-moe-maker validate recipe.yaml     structure only; no GPU, no network
    ms-moe-maker build    recipe.yaml     full pipeline → GGUF
    ms-moe-maker smoke    recipe.yaml     does the artifact generate at all?
    ms-moe-maker eval     recipe.yaml     routing + quality; dead-expert flag
    ms-moe-maker describe                 one line of JSON, zero side effects

Add --json to any build to get JSON Lines events on stdout and prose on
stderr. That is the stream seren-theatre's stagehand consumes, and it is
declared in _describe.EVENTS as a wire contract: unknown event kinds must be
ignored, so adding one is not breaking, removing or renaming one is.

Each stage is overridable in the recipe (custom scripts).  We provide the
floor; if you want to do your own thing you can replace the stage call with
your own script and pass the path in the recipe's `smoke:` or `eval:` dict.

"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from . import _describe as _d
from . import hardware
from .events import Events


def _box_defaults():
    """The layers this install would apply, and what each one sets.

    Reading two small yamls is a READ, not a side effect, so --describe keeps
    its promise. `keys` is provenance: dotted key -> the file that last set it,
    which is the difference between showing a number and explaining it.
    """
    try:
        from . import defaults as _defaults
        resolved, prov, warns = _defaults.resolve()
        digests = _defaults.file_digests()
        layers = [{"label": label, "path": path,
                   "present": path in digests,
                   "sha256": digests.get(path, "")}
                  for label, path in _defaults.layer_paths()]
        return {"layers": layers, "keys": prov,
                "blocks": sorted(resolved), "warnings": list(warns)}
    except Exception:
        return {"layers": [], "keys": {}, "blocks": [], "warnings": []}


def _box_reasoning():
    """Tag styles and model families this install knows about."""
    try:
        from . import reasoning as _rz
        styles, families, warns = _rz.load()
        return {
            "styles": [{"key": k, "name": v.name, "open": v.open,
                        "close": v.close, "interwoven": v.interwoven}
                       for k, v in sorted(styles.items())],
            "families": [{"key": k, "name": v.name, "style": v.style}
                         for k, v in sorted(families.items())],
            "warnings": list(warns),
        }
    except Exception:
        return {"styles": [], "families": [], "warnings": []}


def _box_tiers():
    """Tier names this install offers, floor + whatever the box adds.

    Defensive to a fault because it runs at import time and feeds --describe,
    which promises exit 0 and one line of JSON. A broken defaults file must
    degrade to the floor, never take the contract down with it.
    """
    try:
        from . import defaults as _defaults
        box, _, _ = _defaults.resolve()
        table, _ = hardware.merge_tiers(box.get("tiers"))
        return sorted(table)
    except Exception:
        return sorted(hardware.TIERS)

# ── allocator policy, set BEFORE anything can initialise CUDA ─────────────
#
# On unified memory the caching allocator's RESERVATION is host RAM. Torch's
# default segment policy strands blocks it cannot reuse, and a MoE decode loop
# - many small gathers of varying token counts, per expert, per layer, per
# step - fragments it badly. Measured on a GB10: 106.6 GB reserved to hold
# 6.4 GB of live tensors, a 16x balloon, 100 GB of it doing nothing. On a
# discrete GPU that is invisible; here it is subtracted from the OS, and the
# OOM killer starts reaping 640 kB daemons because they are the only thing it
# can see.
#
# expandable_segments lets a segment grow instead of stranding blocks. It has
# to be in the environment before the first CUDA allocation, which is why it
# lives at import time in the entry point rather than in eval.py.
#
# An existing value always wins: this is a default, not a mandate.
if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def _version() -> str:
    """The installed version, not a literal. This used to be a hardcoded "1.0"
    sitting next to a setuptools-scm dynamic version in pyproject.toml, which
    is two answers to one question."""
    try:
        from ._version import version  # written by setuptools-scm
        return str(version)
    except Exception:
        try:
            from importlib.metadata import version as _v
            return _v("ms-moe-maker")
        except Exception:
            return "0.0.0.dev0"


# ── helpers ──────────────────────────────────────────────────────────────────

def _force_utf8_stdio():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


class _Tee:
    """Write to two streams at once, flushing both."""
    def __init__(self, primary, secondary):
        self._a, self._b = primary, secondary

    def write(self, text):
        self._a.write(text)
        try:
            self._b.write(text)
            self._b.flush()
        except (OSError, ValueError):
            pass
        return len(text)

    def flush(self):
        for s in (self._a, self._b):
            try:
                s.flush()
            except (OSError, ValueError):
                pass


def _corpus_kinds():
    """The registered corpus kinds. Read from the registry, not a literal, so a
    kind published by a plugin shows up in `describe` without an edit here."""
    from . import corpus
    return corpus.names()


# ONE SOURCE OF TRUTH, AND MERGED RATHER THAN RETYPED.
#
# There used to be three lists of verbs that disagreed: this dict said five,
# `_describe.COMMANDS` said three, and the module docstring said "Three verbs".
# _describe is canonical - it is stdlib-only by design so it can answer on a
# half-installed tool, and stagehand reads it to check which contract version
# the thing it forked speaks.
#
# The first fix rebuilt this dict by naming _describe's keys ONE BY ONE, and
# promptly dropped `requires` - a published contract key that release CI
# asserts. Hand-copying a subset of a source of truth is not using the source
# of truth, it is making a second one that starts out correct.
#
# So: start from _describe.DESCRIBE whole, then add the keys that only the
# installed CLI can answer. A key added over there can never be lost here.
DESCRIBE = {
    **_d.DESCRIBE,
    "version": _version(),
    "kinds": _corpus_kinds(),
    "gates": ["auto", "manual", "skip"],
    "templates": ["code", "dnd", "math", "culinary"],
    # THE TIERS THIS INSTALL ACTUALLY HAS, not the ones the source ships. A box
    # can add or redefine a tier from its defaults file, and --describe is how
    # Starwright and Theatre find out what a machine offers - reporting the
    # built-in floor there would describe a different install than the one
    # answering. Reading two small yamls is a read, not a side effect.
    "tiers": _box_tiers(),
    # WHAT THIS BOX PRESETS, so a front-end can show a machine's configuration
    # without re-implementing the merge. Same defensiveness as _box_tiers: a
    # broken file degrades to "nothing configured", never to a broken contract.
    "defaults": _box_defaults(),
    "reasoning": _box_reasoning(),
    "modes": list(_d.EVAL_MODES),
}


def _load_recipe(path, quiet: bool = False, defaults_path=None):
    """Load and validate a recipe file. Returns (Recipe, errs, warns).

    `quiet` exists because under --json stdout belongs to the event stream and
    nothing else may write to it. Prose goes to stderr or nowhere; a stray
    print here would corrupt the very format a consumer is parsing.
    """
    from .recipe import load, validate as validate_recipe

    def out(msg):
        if quiet:
            print(msg, file=sys.stderr)
        else:
            print(msg)

    try:
        rec, parse_warns = load(path, defaults_path=defaults_path)
    except Exception as exc:
        out(f"FAILED to parse {path}: {exc}")
        return None, None, None

    errs, warns = validate_recipe(rec)
    warns = parse_warns + warns

    if errs:
        out(f"\nRecipe has {len(errs)} error(s):")
        for e in errs:
            out(f"  ✗ {e}")
        return None, errs, warns

    return rec, errs, warns


def _build_output_dir(rec) -> str:
    """Find the output directory from the recipe or config."""
    from .config import build_config
    
    try:
        config = build_config(rec)
        return config.output_root
    except Exception:
        return ""


def _find_gguf(output_dir: str):
    """Find the GGUF file in the output directory."""
    from pathlib import Path
    for f in Path(output_dir).rglob("*.gguf"):
        if f.is_file():
            return str(f)
    return ""


# ── commands ─────────────────────────────────────────────────────────────────

DEFAULTS_TEMPLATE_HEADER = """\
# Written by `ms-moe-maker init --defaults-template`.
#
# THIS FILE IS A RECIPE WITH NO EXPERTS. Same keys, same blocks, same `-1`
# sentinels ("you decide"), same typo warnings. Anything you can put in a
# recipe's budget:/corpus:/router:/moe:/eval:/runtime: blocks belongs here too,
# and every recipe on this box inherits it without saying a word.
#
# That is what this is FOR: set a machine up once - for yourself, or for
# somebody you are handing it to - so their recipes stay six lines instead of
# carrying eleven lines they would have to be told about.
#
# Precedence: built-in floor -> packaged defaults.yaml -> THIS FILE ->
# --defaults <path> -> the recipe. The recipe always wins.
#
# `experts:`, `name:` and `template:` are NOT accepted here. Those describe one
# build, not a box. `tiers:` and `models:` are the other way round: box only.
#
# Everything below is commented out. Uncomment what you mean.
"""


def _defaults_template_body() -> str:
    """The starter file. Every line commented; uncommenting beats inventing."""
    return DEFAULTS_TEMPLATE_HEADER + """
# budget:
#   target_steps: 1200        # the biggest lever on wall-clock
#   max_seq_length: 2048
#   lora_r: -1                # -1 = this tier's default

# corpus:
#   min_samples: -1           # floor per expert; rises to meet router_mix_total
#   max_samples: 100000
#   router_mix_total: 16000   # / (batch x accum) = router steps
#   per_repo_cap: 20          # ONE repo must not become the corpus

# router:
#   epochs: 1.0               # the cheapest way to buy router steps
#   batch: 8                  # must be > 1 or the aux loss sees one domain
#   accum: 1

# runtime:
#   hardware_tier: xavier     # see `tiers:` below to add your own
#   llama_cpp: ''             # the path most likely to differ per box

# tools_expert:               # what `tools_expert: true` gets you
#   name: agentcore
#   teacher: Qwen/Qwen2.5-7B-Instruct

# ── BOX ONLY ────────────────────────────────────────────────────────────────
# A recipe may NAME a tier; it may never redefine one, or the same recipe would
# mean different hardware depending on who ran it.

# tiers:
#   spark:
#     default_size: 14B
#     default_lora_r: 96
#   orin_agx:                 # a tier the tool has never heard of
#     like: spark             # inherit the rest, change three things
#     max_vram_gb: 64
#     default_size: 7B
#     default_quant: Q5_K_M

# models:                     # a local mirror, or a house preference
#   "0.5B": /mnt/models/Qwen2.5-Coder-0.5B-Instruct
#   "7B":
#     safe: Qwen/Qwen2.5-Coder-7B
#     abliterated: huihui-ai/Qwen2.5-Coder-7B-Instruct-abliterated
"""


def _write_defaults_template(args) -> int:
    """`init --defaults-template` — the on-ramp for the BOX, not the build.

    Same reasoning as the recipe on-ramp one function down: a newcomer had to
    know the schema before they could type anything, and the only worked
    example was a file in a repo they might never open. Refuses to clobber,
    because the file this overwrites is somebody's machine configuration.
    """
    from . import defaults as _defaults
    target = getattr(args, "output", "") or ""
    if target == "-":
        print(_defaults_template_body())
        return 0
    dest = Path(target or _defaults._user_path()).expanduser()
    if dest.exists() and not args.force:
        print(f"{dest} already exists. Pass --force to overwrite it, or "
              f"--output <path> to write somewhere else.", file=sys.stderr)
        return 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(_defaults_template_body(), encoding="utf-8")
    print(f"wrote {dest}")
    print("Every line is commented out; uncomment what you want this box to "
          "preset.")
    print("`ms-moe-maker validate <recipe>` will then show you which values "
          "came from it.")
    return 0


def _cmd_init(args):
    """Write a starting recipe. The lowest-barrier on-ramp there is.

    The kindness rule this whole tool runs on says: accept a minimum, fill
    sensible defaults, let people who want to twiddle knobs twiddle knobs. That
    rule was true of the PARSER and false of the experience - a newcomer still
    had to know the schema before they could type anything at all, and the
    only worked example was a file in the repo they might never open.

    So this emits a recipe that is already valid, with the optional half
    present but commented out. Uncommenting is a much smaller ask than
    inventing.
    """
    from .template import TEMPLATES, get_template

    if getattr(args, "defaults_template", False):
        return _write_defaults_template(args)

    name = args.template or ""
    if name and name not in TEMPLATES:
        print(f"unknown template {name!r}. Known: "
              f"{', '.join(sorted(TEMPLATES))}", file=sys.stderr)
        return 1

    tpl = get_template(name) if name else None
    experts = (tpl or {}).get("default_experts") or []

    lines = ["# Generated by `ms-moe-maker init`. Everything commented out has",
             "# a sensible default - uncomment only what you want to change.",
             "schema_version: 1"]
    if name:
        lines.append(f"template: {name}")
        lines.append(f"# The {name} template fills in name, base, size and the")
        lines.append("# expert list below. Swap the experts for your own.")
    else:
        lines.append("name: my-moe")

    lines += ["", "experts:"]
    if experts:
        # Serialise the source mapping with the YAML library rather than by
        # hand. The hand-rolled version joined fields with spaces instead of
        # commas and emitted invalid YAML, so `init` produced a recipe that
        # `validate` could not parse - the on-ramp fell over on its first step.
        # Caught by round-tripping init through validate, which is now a test.
        import yaml as _yaml
        for e in experts:
            src = dict(e.get("source", {}))
            flow = _yaml.safe_dump(src, default_flow_style=True,
                                   sort_keys=False).strip().rstrip("\n")
            lines.append(f"  - name: {e.get('name')}")
            lines.append(f"    source: {flow}")
    else:
        lines += [
            "  # At least two. One expert is a dense model with extra steps.",
            "  - name: first",
            "    source: { kind: hf, repo: owner/dataset, text_field: text }",
            "  - name: second",
            "    source: { kind: gh, repo: owner/repo, glob: 'docs/**/*.md' }",
        ]

    lines += [
        "",
        "# size: auto            # auto | 0.5B | 1.5B | 3B | 7B | 14B | 32B",
        "# base: ''              # blank means a supported default for the size",
        "",
        "# tools_expert: true    # add a default MCP/tool-calling expert (kind: synth)",
        "",
        "# runtime:",
        "#   hardware_tier: xavier   # nano | xavier | spark",
        "",
        "# eval:                 # we provide the floor; replace it if you like",
        "#   mode: all           # routing | quality | experts | all",
        "#   dead_threshold: 1.2 # minimum router enrichment before 'dead'",
        "",
        f"# Source kinds available here: {', '.join(_corpus_kinds())}",
        "# Next:  ms-moe-maker validate recipe.yaml",
        "#        ms-moe-maker build recipe.yaml --plan",
        "",
    ]
    text = "\n".join(lines)

    if args.output and args.output != "-":
        out = Path(args.output)
        if out.exists() and not args.force:
            print(f"{out} already exists (use --force to overwrite)",
                  file=sys.stderr)
            return 1
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")
        print(f"  next: ms-moe-maker validate {out}")
    else:
        # Default to stdout so `ms-moe-maker init > recipe.yaml` works and
        # nothing is written without being asked.
        sys.stdout.write(text)
    return 0


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

    from .config import build_config
    from .levers import translate
    from .runner import Runner

    config = build_config(rec, force=args.force, dryrun=args.dryrun)
    translation = translate(rec, force=args.force, dryrun=args.dryrun)

    # TWO IDS, TWO QUESTIONS. recipe_id answers "is this the recipe you sent
    # me"; build_id answers "will my machine build what yours did". They stopped
    # being the same question when defaults moved onto the box.
    try:
        from .config import build_id as _bid
        _build = _bid(config)
    except Exception:
        _build = "?"
    say(f"Ms.MoE — {config.name}  size={config.size}  tier={config.tier}")
    say(f"  ids      recipe {rec.recipe_id()}   build {_build}")
    # WHAT THAT TIER MEANS ON THIS BOX. The name alone stopped being enough the
    # moment a defaults file could redefine one: two machines can both say
    # `tier=spark` and mean different sizes, ranks and quants.
    try:
        from .config import tier_table as _tt
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
        from . import defaults as _dm
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
        from . import preflight as _pf
        from . import stages as _st

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
        for i, (sid, label) in enumerate(_st.plan(config.expert_names, synth), 1):
            say(f"  {i:>2}. {sid:<28} {label}")
        say(f"\n  {len(translation.agreed)} field(s) honoured, "
            f"{len(translation.refusals)} refused. Nothing was run.")
        events.done(ok=True, run_dir=config.output_root, stages_done=0,
                    stages_total=len(_st.plan(config.expert_names, synth)),
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


def _cmd_smoke(args):
    """Smoke-test the GGUF model — proves it generates outside Python."""
    from .export import smoke_gguf
    
    rec, errs, warns = _load_recipe(args.recipe, defaults_path=getattr(args, 'defaults', None))
    if rec is None:
        return 1
    
    from .config import build_config
    config = build_config(rec)

    output_dir = _build_output_dir(rec)
    if not output_dir:
        print("ERROR: could not resolve output directory from recipe.")
        return 1
    
    gguf_path = _find_gguf(output_dir)
    if not gguf_path:
        print(f"ERROR: no GGUF file found in {output_dir}.")
        print(f"  Run 'ms-moe-maker build {args.recipe}' first.")
        return 1
    
    print(f"\n{'=' * 60}")
    print(f"Smoking GGUF: {gguf_path}")
    print(f"{'=' * 60}")
    
    if args.dryrun:
        print(f"[DRY-RUN] Would smoke-test: {gguf_path}")
        return 0
    
    try:
        # THE RECIPE IS THE FLOOR, THE FLAGS OVERRIDE IT. rec.smoke was
        # documented, parsed, and then ignored here - exactly the gap eval had,
        # where a block the README tells people to write reached nothing.
        #
        # llama_cpp_dir is passed because smoke previously searched PATH alone
        # while the build searched the llama.cpp tree, so an ordinary checkout
        # worked during a build and not for the command whose whole job is to
        # run it.
        ok = smoke_gguf(
            gguf_path,
            tokens=args.tokens or rec.smoke.tokens,
            timeout=args.timeout or rec.smoke.timeout,
            prompt=rec.smoke.prompt,
            llama_cpp_dir=config.llama_cpp_dir,
        )
    except Exception as exc:
        print(f"\nSmoke test FAILED: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2
    
    if ok:
        print("\nSmoke test PASSED — model generates text outside Python.")
        return 0
    else:
        print("\nSmoke test FAILED — model did not produce valid output.")
        return 2


def _cmd_eval(args):
    """Routing and/or quality. Never a fabricated number.

    Two bugs lived here. `mode = args.mode` read a dest argparse never created
    (the flag declares dest="eval_mode"), so this raised AttributeError on
    EVERY invocation - the verb had never once run to completion. And the spec
    was hardcoded in the function body, so the recipe's `eval:` block, which
    the README documents and run_eval reads, could not reach it.
    """
    rec, errs, warns = _load_recipe(args.recipe, defaults_path=getattr(args, 'defaults', None))
    if rec is None:
        return 1

    from .config import build_config
    from .eval import run_eval

    config = build_config(rec, force=args.force)

    # The recipe is the floor; --mode overrides it for this one run.
    spec = {
        "script": rec.eval.script,
        "mode": args.eval_mode or rec.eval.mode,
        "held_out_fraction": rec.eval.held_out_fraction,
        "num_samples": rec.eval.num_samples,
        "dead_threshold": rec.eval.dead_threshold,
    }

    print(f"\nEvaluation - mode={spec['mode']}"
          + (f"  (custom script: {spec['script']})" if spec["script"] else ""))

    if args.plan:
        print(f"[plan] would run eval: {spec}")
        return 0

    try:
        report = run_eval(config, spec=spec)
    except Exception as exc:
        print(f"\nEval FAILED: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2

    if not report.ok:
        print(f"\nEval could not run: {report.message}", file=sys.stderr)
        return 2

    _print_eval_report(report)

    # THREE OUTCOMES, THREE EXIT CODES. "We could not measure it" must never
    # share a code with "it passed" - conflating those is exactly what the old
    # proxy scorer did, and why it read as good news for a check that could not
    # fire.
    if report.caveats:
        print("\n[~] READ WITH THIS IN MIND:")
        for c in report.caveats:
            print(f"      - {c}")
    if report.undiscriminating:
        print(f"\n[~] NOT SPECIALISED: {', '.join(report.undiscriminating)}")
        print("      Used, but showing no preference for their own domain.")
        print("      The stitch is fine; this is a router-training result.")
        print("      Fix: more router steps — raise router.epochs (free) or")
        print("      router_mix_total. Measured: corpus quality, domain contrast")
        print("      and expert strength do not move enrichment; the router's own")
        print("      step count does.")
    if report.dead_experts:
        print(f"\n[!] DEAD EXPERTS: {', '.join(report.dead_experts)}")
        return 2
    if report.unmeasured:
        print(f"\n[?] UNMEASURABLE ({len(report.unmeasured)}):")
        for u in report.unmeasured:
            print(f"      - {u}")
        print("\n    Nothing failed. Nothing was proven either.")
        return 3
    print("\n[ok] No dead experts. Every check measured.")
    return 0


def _print_eval_report(report):
    """Print an EvalReport. Experts first, then routing - it is the claim."""
    # EXPERTS BEFORE ROUTING, because it is the question routing makes you ask.
    # Reading "1.00x enrichment" before "the experts are interchangeable" sends
    # you to the router; reading them the other way round does not.
    if report.experts:
        from . import experts as _ex
        rep = _ex.ExpertsReport(
            status=report.experts.get("status", _ex.OK),
            divergence=report.experts.get("divergence", {}),
            pairwise=report.experts.get("pairwise", {}),
            cross_loss=report.experts.get("cross_loss", {}),
            config_audit=report.experts.get("config_audit", {}),
            findings=report.experts.get("findings", []),
            unmeasured=report.experts.get("unmeasured", []))
        print(_ex.format_report(rep))

    routing = report.routing or {}
    experts = routing.get("experts") or {}
    if experts:
        print("\n  ROUTING — P(expert selected | source), all MoE layers pooled")
        print(f"  {'expert':16} {'own':>7} {'others':>8} {'enrich':>9} "
              f"{'top rival':>14} {'share':>7}")
        print(f"  {'-'*16} {'-'*7} {'-'*8} {'-'*9} {'-'*14} {'-'*7}")
        for name, e in sorted(experts.items()):
            flag = ""
            if e.get("outranked"):
                flag = "  OUTRANKED ON ITS OWN GROUND"
            elif e.get("own_is_column_max"):
                flag = "  <- own is top"
            # An abandoned expert's enrichment is one noise divided by
            # another. Printing "2.15x" next to a 0.001 share invites the
            # reader to quote the best-looking number in the table.
            if e.get("enrichment_reliable", True):
                enrich = f"{e['enrichment']:>8.2f}x"
            else:
                enrich = f"{'noise':>9}"
                flag = "  STARVED - enrichment unreadable, read the share"
            print(f"  {name:16} {e['own_share']:>7.3f} {e.get('others_share', 0):>8.3f} "
                  f"{enrich} {str(e.get('top_competitor','')):>14} "
                  f"{e.get('top_competitor_share', 0):>7.3f}{flag}")
        excluded = routing.get("excluded") or []
        if excluded:
            print(f"    {'':16} {'':>7} {'':>8} {'':>9} {'':>14} {'':>7}")
            for name in excluded:
                print(f"  {name:16} {'NOT SCORED':>7} - no held-out rows "
                      f"left after the router mix")
        n = routing.get("named_experts") or 0
        if n:
            width = "" if not excluded else (
                f" (of {n + len(excluded)} experts; {', '.join(excluded)} "
                f"not scored)")
            print(f"\n    own-expert is the column maximum for "
                  f"{routing.get('own_is_max_count', 0)}/{n}{width}")
            print(f"    mean enrichment {routing.get('mean_enrichment', 0):.2f}x"
                  f"   p={routing.get('p_value', 0):.5f} for {n}/{n} by chance")
        js = routing.get("mean_js_bits")
        if js is not None:
            verdict = ("INPUT-BLIND — the router ignores its input entirely"
                       if js < 1e-3 else "routing depends on the input")
            print(f"    mean pairwise JS divergence {js:.4f} bits over "
                  f"{routing.get('moe_layers', 0)} MoE layers — {verdict}")

        # CONFIDENCE SITS NEXT TO JS ON PURPOSE. Saturated-and-blind is a
        # different diagnosis from balanced-and-blind, and share cannot tell
        # them apart: the first is a gate maximising its own output scale
        # (norm_topk_prob=false gives it a free multiplicative gain on a frozen
        # expert), the second is a gate that never left its initialisation.
        # Same enrichment table, opposite fixes.
        conf = routing.get("mean_gate_confidence")
        unif = routing.get("uniform_confidence")
        if conf is not None and unif:
            note = ("  <- SATURATED: the gate is not choosing, it is "
                    "maximising its own output scale" if conf > 0.95 else "")
            print(f"    mean gate confidence {conf:.3f} "
                  f"(uniform would be {unif:.3f}){note}")
    elif routing.get("status") == "unmeasurable":
        print(f"\n  Router discrimination: UNMEASURABLE - {routing.get('reason')}")

    quality = {k: v for k, v in report.stages.items() if not k.startswith("moe/")}
    if quality:
        print("\n  Generation quality (held-out, real generation)")
        print(f"  {'expert':18} {'exact':>7} {'rouge1':>7} {'bleu':>7}  status")
        print(f"  {'-'*18} {'-'*7} {'-'*7} {'-'*7}  {'-'*12}")
        for name, r in sorted(quality.items()):
            print(f"  {name:18} {r.exact_match:>7.3f} {r.rouge1:>7.3f} "
                  f"{r.bleu:>7.3f}  {r.status}")
            moe = report.stages.get(f"moe/{name}")
            if moe is not None:
                print(f"  {'  L moe here':18} {moe.exact_match:>7.3f} "
                      f"{moe.rouge1:>7.3f} {moe.bleu:>7.3f}  {moe.status}")

        # SCORED ON THE ANSWER, NOT THE THINKING. When the base is a reasoning
        # model, say separately how often it actually emitted a think block —
        # "reasons but wrong" and "never reasons" are different failures.
        reasoned_rows = {n: r for n, r in sorted(quality.items())
                         if r.reasoned >= 0}
        if reasoned_rows:
            print("\n  Reasoning (fraction of outputs that emitted a think block)")
            for name, r in reasoned_rows.items():
                flag = "" if r.reasoned > 0.5 else "   <-- does not reliably reason"
                print(f"    {name:16} {r.reasoned:>6.2f}{flag}")

    print(f"\n  {report.message}")


def _cmd_corpus(args):
    """Inspect the corpora on disk. With --prune, PROPOSE a cleaner one.

    PROPOSE, NEVER COMMIT. Without --prune this only measures. With --prune it
    writes a NEW file next to the original and leaves the original untouched,
    because a machine deciding unattended that some of your data should not
    exist is the same shape as a consolidator writing to long-term without a
    human - and this project built a gate to stop exactly that. You read the
    proposal, you point the recipe at the pruned file if you agree, and if you
    disagree nothing has happened.

    Note what prune CANNOT fix: repo dominance on a corpus collected before
    provenance stamping. Those rows are `{"text": ...}` with no `repo` field,
    so the rule that matters most cannot run and says so rather than pruning
    on the two weaker signals and reporting success.
    """
    from . import corpus as corpus_mod
    from . import corpushealth as ch

    rec, errs, _ = _load_recipe(args.recipe, defaults_path=getattr(args, 'defaults', None))
    if rec is None:
        for e in (errs or [f"could not parse {args.recipe}"]):
            print(f"  ✗ {e}")
        return 1

    paths = _corpus_paths(rec)
    if not any(paths.values()):
        print("No corpora on disk for this recipe. Run `build` first.")
        return 3

    findings = 0
    for e in rec.experts:
        path = paths.get(e.name) or ""
        if not path:
            print(f"\n  {e.name}: not collected yet")
            continue
        kind = corpus_mod.get(getattr(e.source, "kind", "")) if e.source else None
        generated = bool(getattr(kind, "generated", False))
        h = ch.inspect(path, generated=generated)
        print()
        print(ch.format_health(h))
        findings += len(h.findings)

        cap = int(getattr(args, "per_repo_cap", 0) or 20)
        if args.prune:
            out_path = path.replace(".jsonl", ".pruned.jsonl")
            pr = ch.write_pruned(path, out_path, per_repo_cap=cap)
            print(f"      wrote {out_path}: kept {pr.keep:,}, dropped {pr.drop:,}")
        else:
            pr = ch.propose_prune(path, per_repo_cap=cap)
            print(f"      --prune would keep {pr.keep:,} and drop {pr.drop:,}")
        for reason, n in pr.reasons.most_common():
            print(f"        {n:>7,}  {reason}")
        for u in pr.unmeasured:
            print(f"        [?] {u}")
        if not args.prune and pr.drop:
            print(f"      (nothing written - re-run with --prune to produce "
                  f"{path.replace('.jsonl', '.pruned.jsonl')})")

    return 0 if not findings else 0


def _cmd_validate(args):
    """Validate recipe structure only — no pipeline, no GPU, no network.

    --json WORKS HERE TOO, and that is the point of the flag. It used to be
    wired into `build` alone, so `ms-moe-maker validate r.yaml --json` was
    accepted by argparse and then printed prose - a machine consumer got an
    empty event stream and no way to tell "valid" from "the flag did nothing".
    A wire format that only some verbs speak is not a wire format.

    Validate has no stages, so the stream is short by nature: started, a
    warning per warning, an error per error, and a terminal done. Terminal is
    the part that matters - a consumer following the stream needs one event
    that means "there will be no more".
    """
    events = Events(enabled=bool(args.json))
    say = events.say if args.json else print

    rec, errs, warns = _load_recipe(args.recipe, quiet=bool(args.json), defaults_path=getattr(args, 'defaults', None))
    if rec is None:
        events.emit("started", recipe=str(args.recipe))
        for e in (errs or []):
            events.error(stage="validate", message=e)
        if errs is None:
            events.error(stage="validate", message=f"could not parse {args.recipe}")
        events.done(ok=False, errors=len(errs or []) or 1, warnings=0)
        return 1

    events.emit("started", recipe=str(args.recipe), recipe_id=rec.recipe_id(),
                name=rec.name, size=rec.size,
                experts=[e.name for e in rec.experts])

    say(f"\n  Recipe: {rec.name or '(auto-filled)'}  [{rec.recipe_id()}]")
    # The recipe id is what you wrote; the build id is what this box will make
    # of it. Printing only the first is how "but it works on mine" happens.
    try:
        from .config import build_config as _bc, build_id as _bid
        say(f"  Build:  {_bid(_bc(rec, dryrun=False))}")
    except Exception:
        pass
    say(f"  Base:   {rec.base or '(auto-filled from tier)'}")
    say(f"  Size:   {rec.size}")
    say(f"  Experts: {[e.name for e in rec.experts]}")
    say(f"  Template: {rec.template or '(none)'}")

    # WHERE EVERY NON-RECIPE VALUE CAME FROM. Defaults live in a file on this
    # box on purpose - so a machine can be set up once for someone else - and
    # the cost of that is a recipe that no longer fully describes its own
    # build. Provenance is what keeps that honest, and validate is the command
    # people run when something is surprising.
    prov = getattr(rec, "defaults_provenance", None) or {}
    if prov:
        say(f"\n  DEFAULTS ({len(prov)} from outside the recipe):")
        for _k, _v in sorted(prov.items()):
            say(f"    {_k:28} <- {_v}")
        # ON THE WIRE NOW. `defaults` is declared in _describe.EVENTS, and the
        # rule there is explicit: a consumer that does not know a kind ignores
        # it, so adding one is additive. It was held back until there was
        # something to say - a vocabulary is easier to add to than to take back.
        events.emit("defaults", provenance=prov,
                    files=dict(getattr(rec, "defaults_digests", None) or {}))

    if warns:
        say(f"\n  WARNINGS ({len(warns)}):")
        for w in warns:
            say(f"    · {w}")
            events.warning(w)
    else:
        say("\n  No warnings.")

    # Refusals are a legitimate answer, not a failure: they are fields the
    # recipe asked for that this build cannot honour. Named on the wire so a
    # consumer can show them without parsing prose.
    from .levers import translate
    refusals = translate(rec).refusals
    if refusals:
        events.refused(refusals)
        say(f"\n  REFUSED ({len(refusals)}):")
        for r in refusals:
            say(f"    ✗ {r}")

    # CORPUS HEALTH, FOR WHATEVER IS ALREADY ON DISK.
    #
    # It belongs here because it is pure stdlib and honours the laptop promise
    # - no torch, no GPU, no network - and because after a build `validate`
    # becomes a re-check you can run in a second. Before a build there is
    # nothing to read, and it says that rather than printing nothing, since a
    # check that vanishes reads like a check that passed.
    findings = _validate_corpora(rec, say, events)

    # `errs` is always empty here - _load_recipe returns rec=None whenever it
    # is not - so printing "0 errors" was theatre. Say what is true.
    say(f"\n  Valid. {len(warns)} warning(s), "
        f"{findings} corpus finding(s).")
    events.done(ok=True, warnings=len(warns), refusals=len(refusals),
                corpus_findings=findings)
    return 0


def _corpus_paths(rec) -> Dict[str, str]:
    """Where this recipe's corpora would live, without building anything."""
    from . import config as cfg_module
    roots = cfg_module.resolve_run_roots(rec)
    data_root = roots["data"]
    out: Dict[str, str] = {}
    for e in rec.experts:
        safe = cfg_module.safe_name(e.name)
        for cand in (f"{data_root}/{safe}_code.jsonl", f"{data_root}/{safe}.jsonl"):
            if os.path.isfile(cand):
                out[e.name] = cand
                break
        else:
            out[e.name] = ""
    return out


def _validate_corpora(rec, say, events) -> int:
    """Report on every corpus that exists. Never builds one."""
    from . import corpus as corpus_mod
    from . import corpushealth as ch

    paths = _corpus_paths(rec)
    if not any(paths.values()):
        say("\n  Corpora: none on disk yet - run `build` first, then "
            "`validate` re-checks them.")
        return 0

    say("\n  CORPUS HEALTH")
    total = 0
    for e in rec.experts:
        path = paths.get(e.name) or ""
        if not path:
            say(f"  {e.name}: not collected yet")
            continue
        kind = corpus_mod.get(getattr(e.source, "kind", "")) if e.source else None
        generated = bool(getattr(kind, "generated", False))
        h = ch.inspect(path, generated=generated)
        say(ch.format_health(h))
        for f in h.findings:
            events.warning(f"corpus/{e.name}: {f}")
        total += len(h.findings)
    return total


# The dispatch table, at module scope so it is one definition rather than a
# local that only main() can see. `describe` is handled before argparse (zero
# side effects, per the Starwright contract) so it has no entry here - which is
# the one asymmetry, and it is asserted in the tests rather than assumed.
COMMAND_HANDLERS = {
    "init": _cmd_init,
    "build": _cmd_build,
    "corpus": _cmd_corpus,
    "smoke": _cmd_smoke,
    "eval": _cmd_eval,
    "validate": _cmd_validate,
}


def main(argv=None):
    _force_utf8_stdio()
    
    # --describe scanned before argparse for zero-side-effects.
    if "--describe" in (argv or sys.argv):
        print(json.dumps(DESCRIBE))
        return 0
    
    ap = argparse.ArgumentParser(
        prog="ms-moe-maker",
        description="Ms.MoE Maker — build targeted Mixtures of Experts from a recipe.",
    )
    ap.add_argument("command", nargs="?", default="build",
                    choices=list(_d.COMMANDS),
                    help="command to run (default: build)")
    ap.add_argument("recipe", nargs="?", help="path to recipe .yaml or .json")
    # TWO MEANINGS, TWO FLAGS. --dryrun used to do both jobs badly: _cmd_build
    # printed a plan and returned 0 before ever calling the pipeline, while
    # run_pipeline's own signature documents dryrun as "run on the smallest
    # rung for structural testing" and config reads MSMOE_DRYRUN the same way.
    # So the flag that was supposed to select a cheap REAL build instead
    # guaranteed no build happened, and the smallest-rung path was unreachable
    # from the CLI entirely.
    #
    #   --plan    resolve everything, run nothing. No torch, no GPU, exit 0.
    #   --dryrun  a real build on the smallest rung. Needs torch, like a build.
    ap.add_argument("--plan", action="store_true",
                    help="resolve config and stages, run nothing (no GPU)")
    # THE FILE THAT MAKES A RECIPE PORTABLE AGAIN. Defaults normally come from
    # this box (~/.msmoe/defaults.yaml), which is the point - but it also means
    # the same recipe can build differently in two places. Point at an explicit
    # file to reproduce someone else's run, or to pin a build in CI.
    ap.add_argument("--defaults-template", action="store_true",
                    dest="defaults_template",
                    help="init: write a commented defaults file for this box "
                         "(default: ~/.msmoe/defaults.yaml)")
    ap.add_argument("--defaults", metavar="PATH", default=None,
                    help="defaults file to layer under the recipe "
                         "(default: packaged, then ~/.msmoe/defaults.yaml)")
    ap.add_argument("--offline", action="store_true",
                    help="skip reachability checks (no network calls)")
    ap.add_argument("--dryrun", action="store_true",
                    help="real build on the smallest rung (still needs torch)")
    ap.add_argument("--force", action="store_true",
                    help="redo existing artifacts")
    # THE FLAG THAT WAS MISSING. _describe.EVENTS declares a JSON Lines wire
    # vocabulary and calls renaming an event a breaking change - but there was
    # no --json argument at all, so argparse rejected it with exit 2. Since
    # seren-theatre's stagehand forks the literal documented command
    # `ms-moe-maker build <recipe> --json`, every Theatre-driven build died at
    # argument parsing. Events was even imported here and never used.
    ap.add_argument("--json", action="store_true",
                    help="JSON Lines events on stdout, prose on stderr")
    ap.add_argument("--pipeline", dest="pipeline",
                    help="fork this legacy pipeline script instead of the "
                         "in-package builder")
    ap.add_argument("--python", dest="python", default=None,
                    help="interpreter that runs the pipeline (default: ours). "
                         "The trainer lives in whatever venv has torch; this "
                         "CLI does not have to.")
    
    # Smoke args
    # Default 0, not the real default: it is the only way to tell "the user
    # did not say" from "the user asked for the same number", and the recipe
    # has to win when they did not say.
    ap.add_argument("--tokens", type=int, default=0,
                    help="smoke test token count (default: the recipe's, or 48)")
    ap.add_argument("--timeout", type=int, default=0,
                    help="smoke timeout in seconds (default: the recipe's, or 300)")
    
    # init args
    ap.add_argument("--template", dest="template", default="",
                    help="start from a template (see `describe`)")
    ap.add_argument("-o", "--output", dest="output", default="-",
                    help="write here instead of stdout")

    # Eval args
    # SLICED BY QUESTION, NOT BY MODEL. This used to be expert|moe|all, which
    # asks "which model do I run" - but the two things worth knowing are
    # separate questions: does the router prefer each expert on its own ground
    # (routing, the dead-expert claim), and does the thing answer well
    # (quality, which needs an answer key only the corpus author has).
    # Default is empty so the recipe's eval.mode wins unless overridden here.
    ap.add_argument("--prune", action="store_true",
                    help="corpus: WRITE a pruned copy next to the original "
                         "(never in place). Without it, only propose.")
    ap.add_argument("--per-repo-cap", dest="per_repo_cap", type=int, default=0,
                    help="corpus: max docs per repo when pruning (default 20)")
    ap.add_argument("--mode", dest="eval_mode",
                    choices=list(_d.EVAL_MODES), default="",
                    help="eval mode (default: the recipe's eval.mode, or all)")
    
    a = ap.parse_args(argv)
    
    if a.command == "describe":
        print(json.dumps(DESCRIBE))
        return 0
    
    # init invents a recipe rather than reading one, so it is the other verb
    # that does not want a path.
    if not a.recipe and a.command not in ("describe", "init"):
        ap.error(f"'{a.command}' requires a recipe path")
    
    handler = COMMAND_HANDLERS.get(a.command)
    if handler is None:
        ap.error(f"unknown command: {a.command}")
    
    return handler(a)


if __name__ == "__main__":
    raise SystemExit(main())
