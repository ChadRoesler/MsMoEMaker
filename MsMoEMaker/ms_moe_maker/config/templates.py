"""Recipe templates — pre-built configs for common use cases.

Users pick a template and tweak:  they get a working recipe with zero
boilerplate.  The template fills name, base, size, budget, moe, gates,
runtime, and expert sources so the pipeline can run immediately.

Templates are just dicts that get merged into the recipe before parsing.
A user can always override individual fields after loading.

  dnd      — Dungeons & Dragons lookup (monster manual, DMG, PHB).
  math     — Math problem solver (textbook, exercises, solutions).
  culinary — Recipe assistant (ingredients, techniques, cuisines).
  code     — Code specialist (codeparrot / code datasets, like the original).

Each template declares:
  - preferred_size        — auto-default for model size
  - base_hint             — suggested base model prefix
  - default_tier          — hardware tier hint
  - default_experts       — example expert list with sources
  - default_budget        — target_steps, max_seq_length, etc.
  - default_moe           — experts_per_tok, dense_layers, etc.
  - default_eval          — eval config for post-build checking
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# ── Template registry ──────────────────────────────────────────────────────────

TEMPLATES: Dict[str, Dict[str, Any]] = {
    "code": {
        "name": "code-specialist-moe",
        "preferred_size": "3B",
        "base_hint": "Qwen/Qwen2.5-Coder",
        "default_tier": "spark",
        "default_experts": [
            {
                "name": "python",
                "source": {"kind": "stack", "language": "Python"},
            },
            {
                "name": "csharp",
                "source": {"kind": "stack", "language": "C#"},
            },
        ],
        "default_budget": {
            "target_steps": 1200,
            "max_seq_length": 4096,
            "per_device_batch": 4,
            "grad_accum": 2,
            "warmup_ratio": 0.05,
        },
        "default_moe": {
            "experts_per_tok": 2,
            "norm_topk_prob": True,
            "shared_expert_width": 1,
            "shared_expert_gate_fill": 0.02,
            "dense_layers": "auto",
        },
        "default_gates": {
            "base_evals": "auto",
            "main_evals": "auto",
        },
        "default_runtime": {
            "precision": "float16",
            "load_in_4bit": False,
        },
        "default_eval": {
            "held_out_fraction": 0.1,
            "num_samples": 20,
            "dead_threshold": 1.2,
        },
    },
    "dnd": {
        "name": "dnd-lookup-moe",
        "preferred_size": "0.5B",
        "base_hint": "Qwen/Qwen2.5",
        "default_tier": "nano",
        "default_experts": [
            {
                "name": "monster_manual",
                "source": {
                    "kind": "hf",
                    "repo": "PleiaSys/DnD-MonsterManual",
                    "text_field": "text",
                },
            },
            {
                "name": "players_handbook",
                "source": {
                    "kind": "hf",
                    "repo": "PleiaSys/DnD-PlayerHandbook",
                    "text_field": "text",
                },
            },
            {
                "name": "dm_guide",
                "source": {
                    "kind": "hf",
                    "repo": "PleiaSys/DnD-DMG",
                    "text_field": "text",
                },
            },
        ],
        "default_budget": {
            "target_steps": 500,
            "max_seq_length": 2048,
            "per_device_batch": 1,
            "grad_accum": 4,
            "warmup_ratio": 0.05,
        },
        "default_moe": {
            "experts_per_tok": 2,
            "norm_topk_prob": True,
            "shared_expert_width": 1,
            "shared_expert_gate_fill": 0.02,
            "dense_layers": "auto",
        },
        "default_gates": {
            "base_evals": "auto",
            "main_evals": "skip",
        },
        "default_runtime": {
            "precision": "float16",
            "load_in_4bit": True,
        },
        "default_eval": {
            "held_out_fraction": 0.1,
            "num_samples": 10,
            "dead_threshold": 1.2,
        },
    },
    "math": {
        "name": "math-solver-moe",
        "preferred_size": "1.5B",
        "base_hint": "Qwen/Qwen2.5",
        "default_tier": "spark",
        "default_experts": [
            {
                "name": "arithmetic",
                "source": {
                    "kind": "hf",
                    "repo": "open-webmath/arithmetic",
                    "text_field": "text",
                },
            },
            {
                "name": "algebra",
                "source": {
                    "kind": "hf",
                    "repo": "open-webmath/algebra",
                    "text_field": "text",
                },
            },
            {
                "name": "geometry",
                "source": {
                    "kind": "hf",
                    "repo": "open-webmath/geometry",
                    "text_field": "text",
                },
            },
            {
                "name": "word_problems",
                "source": {
                    "kind": "hf",
                    "repo": "open-webmath/word_problems",
                    "text_field": "text",
                },
            },
        ],
        "default_budget": {
            "target_steps": 800,
            "max_seq_length": 2048,
            "per_device_batch": 2,
            "grad_accum": 4,
            "warmup_ratio": 0.05,
        },
        "default_moe": {
            "experts_per_tok": 2,
            "norm_topk_prob": True,
            "shared_expert_width": 1,
            "shared_expert_gate_fill": 0.02,
            "dense_layers": "auto",
        },
        "default_gates": {
            "base_evals": "auto",
            "main_evals": "auto",
        },
        "default_runtime": {
            "precision": "float16",
            "load_in_4bit": False,
        },
        "default_eval": {
            "held_out_fraction": 0.15,
            "num_samples": 20,
            "dead_threshold": 1.2,
        },
    },
    "culinary": {
        "name": "culinary-assistant-moe",
        "preferred_size": "1.5B",
        "base_hint": "Qwen/Qwen2.5",
        "default_tier": "spark",
        "default_experts": [
            {
                "name": "ingredients",
                "source": {
                    "kind": "hf",
                    "repo": "mrmckay/recipes_europress",
                    "text_field": "text",
                },
            },
            {
                # A PLACEHOLDER PATH, NOT AN EMPTY ONE.
                #
                # This was `"path": ""` with a "user fills this" comment, which
                # meant `init --template culinary` emitted a recipe that failed
                # validation immediately: "source.kind=local needs a path". A
                # template whose output cannot validate is a broken on-ramp,
                # and it fails at the worst moment - the user's first command.
                #
                # A named path validates (structure is right) and then
                # PREFLIGHT reports it does not exist, with the remedy. That is
                # the correct division: validate answers "is this recipe
                # well-formed", preflight answers "will it run on this box".
                "name": "techniques",
                "source": {
                    "kind": "local",
                    "path": "./culinary_techniques",
                    "glob": "**/*.txt",
                },
            },
            {
                "name": "cuisines",
                "source": {
                    "kind": "hf",
                    "repo": "mrmckay/recipes_europress",
                    "text_field": "text",
                },
            },
        ],
        "default_budget": {
            "target_steps": 800,
            "max_seq_length": 2048,
            "per_device_batch": 2,
            "grad_accum": 4,
            "warmup_ratio": 0.05,
        },
        "default_moe": {
            "experts_per_tok": 2,
            "norm_topk_prob": True,
            "shared_expert_width": 1,
            "shared_expert_gate_fill": 0.02,
            "dense_layers": "auto",
        },
        "default_gates": {
            "base_evals": "auto",
            "main_evals": "skip",
        },
        "default_runtime": {
            "precision": "float16",
            "load_in_4bit": False,
        },
        "default_eval": {
            "held_out_fraction": 0.1,
            "num_samples": 10,
            "dead_threshold": 1.2,
        },
    },
}


def get_template(name: str) -> Optional[Dict[str, Any]]:
    """Return the template dict for a template name, or None if unknown."""
    if name not in TEMPLATES:
        return None
    return dict(TEMPLATES[name])


def apply_template(recipe: Dict[str, Any], template_name: str) -> Dict[str, Any]:
    """Merge a template into a recipe dict.

    Template fields fill in wherever the recipe dict is empty / missing.
    The recipe's own values always win.
    """
    tpl = get_template(template_name)
    if tpl is None:
        raise ValueError(
            f"unknown template {template_name!r}. "
            f"Known: {', '.join(sorted(TEMPLATES))}")

    merged = dict(recipe)

    # A TEMPLATE'S OWN METADATA IS NOT RECIPE CONTENT.
    #
    # This used to copy `preferred_size`, `default_tier`, `default_moe`,
    # `default_runtime` and friends straight into the merged dict as TOP-LEVEL
    # keys - and then also read them to fill the real fields. The originals
    # were never removed, so the recipe carried both the answer and the note
    # the answer came from, and parse() then reported every leftover as
    # "X is not a known top-level key - IGNORED".
    #
    # The effect was that `ms-moe-maker init --template dnd` produced a file
    # which validated with TEN warnings, none of which the user had caused or
    # could act on. A brand-new recipe greeting its author with a list of
    # complaints about itself is a bad first thirty seconds, and it teaches
    # people to ignore warnings - which is expensive later, when one matters.
    #
    # So: translate into the real fields, then keep nothing that is not part
    # of the recipe schema.

    if "name" in tpl and "name" not in merged:
        merged["name"] = tpl["name"]

    # preferred_size is the template saying what size it is FOR. Its home in a
    # recipe is `size`.
    if "preferred_size" in tpl and "size" not in merged:
        merged["size"] = tpl["preferred_size"]

    # Experts — template experts are defaults if the recipe has none.
    if "experts" not in merged and "default_experts" in tpl:
        merged["experts"] = tpl["default_experts"]

    # Nested blocks — MERGED PER KEY, not clobbered wholesale. `template: code`
    # plus `budget: {target_steps: 300}` used to drop the template's
    # max_seq_length 4096 and fall to the dataclass 2048, silently halving
    # tokens/expert. The template's block is the floor; each recipe key wins
    # on its own.
    for key in ("budget", "moe", "gates", "runtime", "eval", "smoke"):
        tpl_key = f"default_{key}"
        if tpl_key in tpl:
            merged[key] = {**dict(tpl[tpl_key]), **(merged.get(key) or {})}

    # default_tier is the hardware tier, which lives at runtime.hardware_tier.
    if "default_tier" in tpl:
        runtime = dict(merged.get("runtime") or {})
        runtime.setdefault("hardware_tier", tpl["default_tier"])
        merged["runtime"] = runtime

    # base_hint is advisory and consumed by config, not part of the schema.
    if "base_hint" in tpl and "_base_hint" not in merged:
        merged["_base_hint"] = tpl["base_hint"]

    # Belt and braces: strip any template-internal key that reached the dict,
    # including from a recipe that named one by hand.
    for internal in ("preferred_size", "base_hint", "default_tier",
                     "default_experts", "default_budget", "default_moe",
                     "default_gates", "default_runtime", "default_eval",
                     "default_smoke"):
        merged.pop(internal, None)

    return merged


def describe_templates() -> Dict[str, Any]:
    """Return a summary of all available templates."""
    result = {}
    for name, tpl in TEMPLATES.items():
        result[name] = {
            "name": name,
            "preferred_size": tpl.get("preferred_size", "?"),
            "default_tier": tpl.get("default_tier", "?"),
            "expert_count": len(tpl.get("default_experts", [])),
            "experts": [e["name"] for e in tpl.get("default_experts", [])],
            "description": _template_description(name),
        }
    return result


def _template_description(name: str) -> str:
    """Human-readable description of a template."""
    descriptions = {
        "code": "Multi-language code specialist (Python, C#). Built for coding tasks, inspired by the original fraunkenstein pipeline.",
        "dnd": "Dungeons & Dragons lookup MoE. Monster Manual, Player's Handbook, and DMG specialists. For TTRPG sessions and reference.",
        "math": "Math problem solver with domain specialists (arithmetic, algebra, geometry, word problems).",
        "culinary": "Recipe and cooking assistant with ingredient, technique, and cuisine specialists.",
    }
    return descriptions.get(name, "Unknown template.")
