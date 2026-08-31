"""`ms-moe-maker init` - write a starting recipe or a defaults file."""
from __future__ import annotations

import sys
from pathlib import Path

from ._common import _corpus_kinds


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

# reasoning_expert:           # what `reasoning_expert: true` gets you
#   name: deliberation        #   the teacher must ITSELF reason, or every
#   teacher: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   # trace is rejected

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
    from ..config import defaults as _defaults
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
    from ..config.templates import TEMPLATES, get_template

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
        "# reasoning_expert: true  # add a default reasoning specialist. Its corpus",
        "#                         #   spans the OTHER experts' domains, so it learns",
        "#                         #   the register of deliberation instead of one",
        "#                         #   subject. Do NOT put `reasoning: true` on a",
        "#                         #   domain expert to get this - that spends most",
        "#                         #   of that expert's token budget on prose.",
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
