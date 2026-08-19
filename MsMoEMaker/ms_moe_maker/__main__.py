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
from .events import Events


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


# ONE SOURCE OF TRUTH. There used to be three lists of verbs that disagreed:
# this dict said five, `_describe.COMMANDS` said three, and the module
# docstring above said "Three verbs: build / smoke / eval". _describe is the
# canonical one - it is stdlib-only by design so it can answer on a
# half-installed tool, and its docstring says stagehand checks it to see which
# contract version the thing it forked speaks. A front-end trusting it would
# have concluded `eval` and `smoke` did not exist.
#
# The integration test named test_describe_commands_are_current asserted this
# dict against a hardcoded literal in its own body, so it agreed with the wrong
# copy and stayed green through the whole drift.
DESCRIBE = {
    "name": _d.NAME,
    "version": _version(),
    "kinds": ["hf", "stack", "synth", "local"],
    "gates": ["auto", "manual", "skip"],
    "templates": ["code", "dnd", "math", "culinary"],
    "tiers": ["nano", "xavier", "spark"],
    "commands": list(_d.COMMANDS),
    "events": list(_d.EVENTS),
    "modes": ["routing", "quality", "all"],
    "manifest_schema_version": _d.DESCRIBE["manifest_schema_version"],
    "recipe_schema_version": _d.DESCRIBE["recipe_schema_version"],
    "description": _d.DESCRIPTION,
}


def _load_recipe(path):
    """Load and validate a recipe file. Returns (Recipe, errs, warns) or exits."""
    from .recipe import load, validate as validate_recipe
    
    try:
        rec, parse_warns = load(path)
    except Exception as exc:
        print(f"FAILED to parse {path}: {exc}")
        return None, None, None
    
    errs, warns = validate_recipe(rec)
    warns = parse_warns + warns
    
    if errs:
        print(f"\nRecipe has {len(errs)} error(s):")
        for e in errs:
            print(f"  ✗ {e}")
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
        "# runtime:",
        "#   hardware_tier: xavier   # nano | xavier | spark",
        "",
        "# eval:                 # we provide the floor; replace it if you like",
        "#   mode: all           # routing | quality | all",
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


def _corpus_kinds():
    from . import corpus
    return corpus.names()


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
    rec, errs, warns = _load_recipe(args.recipe)
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

    say(f"Ms.MoE — {config.name}  size={config.size}  tier={config.tier}")
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

        checks = _pf.run(config, rec, offline=True, need_exporter=True)
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
    return runner.run()


def _cmd_smoke(args):
    """Smoke-test the GGUF model — proves it generates outside Python."""
    from .export import smoke_gguf
    
    rec, errs, warns = _load_recipe(args.recipe)
    if rec is None:
        return 1
    
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
        ok = smoke_gguf(
            gguf_path,
            tokens=args.tokens,
            timeout=args.timeout,
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
    rec, errs, warns = _load_recipe(args.recipe)
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
    """Print an EvalReport. Routing first - it is the claim."""
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
            print(f"  {name:16} {e['own_share']:>7.3f} {e.get('others_share', 0):>8.3f} "
                  f"{e['enrichment']:>8.2f}x {str(e.get('top_competitor','')):>14} "
                  f"{e.get('top_competitor_share', 0):>7.3f}{flag}")
        n = routing.get("named_experts") or 0
        if n:
            print(f"\n    own-expert is the column maximum for "
                  f"{routing.get('own_is_max_count', 0)}/{n}")
            print(f"    mean enrichment {routing.get('mean_enrichment', 0):.2f}x"
                  f"   p={routing.get('p_value', 0):.5f} for {n}/{n} by chance")
        js = routing.get("mean_js_bits")
        if js is not None:
            verdict = ("INPUT-BLIND — the router ignores its input entirely"
                       if js < 1e-3 else "routing depends on the input")
            print(f"    mean pairwise JS divergence {js:.4f} bits over "
                  f"{routing.get('moe_layers', 0)} MoE layers — {verdict}")
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

    print(f"\n  {report.message}")


def _cmd_validate(args):
    """Validate recipe structure only — no pipeline, no GPU needed."""
    rec, errs, warns = _load_recipe(args.recipe)
    if rec is None:
        return 1
    
    print(f"\n  Recipe: {rec.name or '(auto-filled)'}  [{rec.recipe_id()}]")
    print(f"  Base:   {rec.base or '(auto-filled from tier)'}")
    print(f"  Size:   {rec.size}")
    print(f"  Experts: {[e.name for e in rec.experts]}")
    print(f"  Template: {rec.template or '(none)'}")
    
    if warns:
        print(f"\n  WARNINGS ({len(warns)}):")
        for w in warns:
            print(f"    · {w}")
    else:
        print("\n  ✓ No warnings.")
    
    # `errs` is always empty here - _load_recipe returns rec=None whenever it
    # is not, so this line is only ever reached on a valid recipe. Printing
    # "0 errors" was theatre; say what is actually true.
    print(f"\n  Valid. {len(warns)} warning(s).")
    return 0


# ── main ──────────────────────────────────────────────────────────────────────

# The dispatch table, at module scope so it is one definition rather than a
# local that only main() can see. `describe` is handled before argparse (zero
# side effects, per the Starwright contract) so it has no entry here - which is
# the one asymmetry, and it is asserted in the tests rather than assumed.
COMMAND_HANDLERS = {
    "init": _cmd_init,
    "build": _cmd_build,
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
    # rung for structural testing" and config reads FRAUNK_DRYRUN the same way.
    # So the flag that was supposed to select a cheap REAL build instead
    # guaranteed no build happened, and the smallest-rung path was unreachable
    # from the CLI entirely.
    #
    #   --plan    resolve everything, run nothing. No torch, no GPU, exit 0.
    #   --dryrun  a real build on the smallest rung. Needs torch, like a build.
    ap.add_argument("--plan", action="store_true",
                    help="resolve config and stages, run nothing (no GPU)")
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
    ap.add_argument("--tokens", type=int, default=48,
                    help="smoke test token count (default: 48)")
    ap.add_argument("--timeout", type=int, default=300,
                    help="smoke test timeout in seconds (default: 300)")
    
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
