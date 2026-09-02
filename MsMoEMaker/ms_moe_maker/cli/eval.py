"""`ms-moe-maker eval` - routing and quality; the dead-expert flag."""
from __future__ import annotations

import sys

from ._common import _load_recipe


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

    from ..config.pipeline import build_config
    from ..eval import run_eval

    config = build_config(rec, force=args.force)

    # The recipe is the floor; --mode overrides it for this one run.
    # held_out_fraction comes from the RESOLVED config (clamped once, in
    # build_config) - the raw recipe value used to reach run_eval unclamped,
    # so the router trained on one split while eval re-split at another.
    spec = {
        "script": rec.eval.script,
        "mode": args.eval_mode or rec.eval.mode,
        "held_out_fraction": config.eval_held_out_fraction,
        "num_samples": rec.eval.num_samples,
        "dead_threshold": rec.eval.dead_threshold,
        # -1 = you decide; run_eval resolves it from whether this run writes
        # thinking traces (pipeline.eval_max_new_tokens). Passed RAW, not
        # resolved here, so the sentinel survives to the one place that knows
        # the run's shape.
        "max_new_tokens": rec.eval.max_new_tokens,
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

    # PERSIST BEFORE PRINTING, and the order is the point. `_print_eval_report`
    # is two hundred lines of formatting over a dozen optional sub-tables, and
    # an exception in any one of them would take the whole measurement with it:
    # forty minutes of generation lost because a table header could not render.
    # The numbers reach disk first; the pretty version is a convenience laid on
    # top of an artifact that already exists.
    #
    # INTO THE RUN DIRECTORY, BESIDE THE MANIFEST AND NOT INTO IT. `eval` is
    # deliberately outside the build - run/stages.py spells out why a model
    # must not grade itself as part of being built - so it must not write the
    # manifest either, which the builder owns. It drops its own file and lets a
    # reader find it, the same bargain the GGUF and the smoke-pass marker make.
    from pathlib import Path
    from ..config.pipeline import build_id as _build_id
    from ..eval.harness import EVAL_REPORT_NAME, save_eval_report

    saved = Path(config.output_root) / EVAL_REPORT_NAME
    try:
        save_eval_report(report, saved, build_id=_build_id(config))
        print(f"\n  eval report -> {saved}")
    except OSError as exc:
        # A read-only or full run directory must not turn a measurement that
        # SUCCEEDED into a command that failed. Say so and carry on to the
        # numbers, which are still correct and still worth reading.
        print(f"\n[warn] could not write the eval report to {saved}: {exc}",
              file=sys.stderr)

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
        from ..train import experts as _ex
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
            hits = routing.get('own_is_max_count', 0)
            print(f"\n    own-expert is the column maximum for "
                  f"{hits}/{n}{width}")
            # SAY WHAT THE p IS FOR, AND LET THE PROBE SAY IT. This line used
            # to read "p=... for {n}/{n} by chance" no matter what `hits` was,
            # so one expert of five winning its column still printed the
            # all-five-of-five significance figure under a failing table. The
            # probe now reports the probability of the event it measured and
            # names it; this only echoes both.
            p = routing.get("p_value")
            event = (routing.get("p_value_event")
                     or f"at least {hits} of {n} by chance")
            tail = (f"   p={p:.5f} for {event}"
                    if isinstance(p, (int, float)) and not isinstance(p, bool)
                    else "   (the probe reported no p-value)")
            print(f"    mean enrichment {routing.get('mean_enrichment', 0):.2f}x"
                  f"{tail}")
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
            # SATURATION IS RELATIVE TO 1/K, NOT TO 1.0.
            # `conf` is the MEAN of the K top softmax probabilities, and those
            # K values sum to at most 1 - so the mean can never exceed 1/K.
            # A fixed `conf > 0.95` was therefore unreachable for any
            # experts_per_tok >= 2 (ceiling 0.50 at K=2), which silently
            # switched off the one diagnosis that distinguishes
            # saturated-and-blind from balanced-and-blind. That is the
            # norm_topk_prob=false failure mode, and share and JS cannot tell
            # them apart - which is the entire reason this column exists.
            k = int(routing.get("top_k") or 1)
            ceiling = 1.0 / max(k, 1)
            note = ("  <- SATURATED: the gate is not choosing, it is "
                    "maximising its own output scale"
                    if conf > 0.95 * ceiling else "")
            print(f"    mean gate confidence {conf:.3f} "
                  f"(uniform would be {unif:.3f}, top-{k} maximum is "
                  f"{ceiling:.3f}){note}")

        # DOES THE GATE HAND OFF AT `</think>`? The pooled table above cannot
        # say - it averages the whole sequence, so a relay and a duet print the
        # same row. THE DELTA IS THE FINDING, so the delta gets a column of its
        # own rather than leaving the reader to subtract two share tables.
        # Absent whenever nothing sampled had a closed think block.
        segs = routing.get("think_segments") or {}
        for src, seg in sorted(segs.items()):
            print(f"\n  ROUTING INSIDE vs AFTER <think> — {src} "
                  f"({seg.get('samples', 0)} sampled rows with a closed block)")
            print(f"    {'expert':16} {'in think':>9} {'after':>9} {'delta':>9}")
            print(f"    {'-'*16} {'-'*9} {'-'*9} {'-'*9}")
            for name in sorted(seg.get("delta") or {}):
                print(f"    {name:16} {seg['think'][name]:>9.3f} "
                      f"{seg['after'][name]:>9.3f} {seg['delta'][name]:>+9.3f}")
            if seg.get("verdict") == "relay":
                print(f"    -> RELAY: {seg['swing_to']} takes "
                      f"{seg['swing']:+.3f} more of the selection slots")
                print(f"       inside the block than after it; "
                      f"{seg['yields_to']} picks it up on the other side.")
            elif seg.get("verdict") == "duet":
                print("    -> DUET: nothing swings more than 0.05 of the "
                      "slots at the tag boundary — routing does not hand off "
                      "at </think>.")
        for src, why in sorted((routing.get("think_segment_errors") or {}).items()):
            print(f"    (no think-block segmentation for {src}: {why} — a "
                  f"missing row here is a tokenizer limit, not an absence of "
                  f"think blocks)")
    elif routing.get("status") == "unmeasurable":
        print(f"\n  Router discrimination: UNMEASURABLE - {routing.get('reason')}")

    quality = {k: v for k, v in report.stages.items() if not k.startswith("moe/")}
    if quality:
        # PRINT THE DENOMINATOR. It existed only in `note`, which this
        # function never printed and detect_dead_experts overwrote, so a row
        # averaged over 3 of 20 rows and a row averaged over all 20 printed
        # identically. `!` marks a sample too thin to compare with a full one.
        def _thin(r):
            return bool(r.attempted_samples) and (
                r.scored_samples < 5
                or r.scored_samples * 2 < r.attempted_samples)

        def _n(r):
            return (f"{r.scored_samples}/{r.attempted_samples}"
                    if r.attempted_samples else "-")

        # `reasoned` IS A QUALITY COLUMN, NOT A FOOTNOTE. It lived in a
        # separate block under the table, which is where a number goes to be
        # read on its own - and on its own it says nothing. It only becomes a
        # diagnosis beside the routing enrichment, so it sits in the same row
        # as the scores it qualifies. '-' rather than 0.00 because a
        # non-reasoning run did not score zero, it was never asked.
        def _r(r):
            return f"{r.reasoned:8.2f}" if r.reasoned >= 0 else f"{'-':>8}"

        def _flags(r):
            out = "  ! thin sample" if _thin(r) else ""
            if 0 <= r.reasoned <= 0.5:
                out += "  ! does not reliably reason"
            return out

        print("\n  Generation quality (held-out, real generation)")
        print(f"  {'expert':18} {'exact':>7} {'rouge1':>7} {'bleu':>7} "
              f"{'reasoned':>8} {'scored':>9}  status")
        print(f"  {'-'*18} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*9}  {'-'*12}")
        thin_rows = []
        for name, r in sorted(quality.items()):
            print(f"  {name:18} {r.exact_match:>7.3f} {r.rouge1:>7.3f} "
                  f"{r.bleu:>7.3f} {_r(r)} {_n(r):>9}  {r.status}"
                  f"{_flags(r)}")
            if _thin(r):
                thin_rows.append(name)
            moe = report.stages.get(f"moe/{name}")
            if moe is not None:
                print(f"  {'  L moe here':18} {moe.exact_match:>7.3f} "
                      f"{moe.rouge1:>7.3f} {moe.bleu:>7.3f} "
                      f"{_r(moe)} {_n(moe):>9}  {moe.status}"
                      f"{_flags(moe)}")
        if thin_rows:
            print(f"    ! {', '.join(thin_rows)}: most rows drawn could not be "
                  f"scored (a `text` row needs 4+ lines to split into a "
                  f"prompt and a reference). Not comparable with a full row.")

        # SCORED ON THE ANSWER, NOT THE THINKING — and `reasoned` says how
        # often there was a thinking block to separate out at all. Say what the
        # column MEANS and what it is read against, because the number alone
        # supports the wrong conclusion: low reasoned looks like a bad model,
        # while low reasoned NEXT TO high routing enrichment is a specific,
        # predicted failure of putting reasoning in a routed FFN expert. The
        # harness raises that one as a caveat; this is the legend for it.
        if any(r.reasoned >= 0 for r in report.stages.values()):
            print("    reasoned = share of outputs that opened AND closed a "
                  "think block.")
            print("      It is the DISCIPLINE check, and it is read against "
                  "routing: high enrichment")
            print("      with low reasoned means the register of deliberation "
                  "without the structure.")

    print(f"\n  {report.message}")
