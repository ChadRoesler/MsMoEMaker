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
