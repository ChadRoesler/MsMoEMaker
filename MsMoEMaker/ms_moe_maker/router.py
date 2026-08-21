"""Router training — train ONLY the MoE router weights.

The specialists are frozen.  Only the `*.mlp.gate.weight` parameters (the
TopKRouter gates) are trainable — roughly 1.2 M params at 0.5B.

Uses a stratified data mix: each expert's corpus contributes a proportional
quota.  Agentcore gets AGENT_MIX_FRACTION of the total; code experts split
the rest equally.
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import Dict, List, Optional

from . import config as cfg_module
from . import stages as st


def router_dir(config) -> str:
    return f"{config.output_root}/" + st.ARTIFACTS[st.ROUTER]


def router_is_done(config) -> bool:
    """Is the router-trained MoE already on disk? See finetune.specialist_is_done."""
    if config.force:
        return False
    return os.path.exists(os.path.join(router_dir(config), "config.json"))


def train_router(config, final_dir: str, safe_names: List[str],
                 expert_corpus_paths: Dict[str, str]) -> str:
    """Train the MoE router on a stratified mix of expert corpora.

    The specialists are loaded frozen. Only the router gate weights train.
    Saves to OUTPUT_ROOT/moe_trained.

    `expert_corpus_paths` maps expert name -> the JSONL CORPUS FILE that expert
    was trained on. It was called `expert_paths`, and the builder passed
    `specialist_dirs` - the MODEL directories - into it. Two different things
    with names close enough to slide past a reader, and the mistake surfaced
    ten minutes into a run as

        IsADirectoryError: [Errno 21] Is a directory: '.../specialist_python'

    which names the symptom and not one word of the cause. Renamed so the call
    site has to say which it means, and checked FIRST so a wrong one is
    refused immediately with the reason rather than at the open().
    """
    # ARGUMENTS BEFORE IMPORTS. A caller's mistake should cost nothing to
    # detect. Checked here rather than lower down because below this line is
    # `import torch` and a multi-gigabyte model load - so a guard placed after
    # them is unreachable on a box without a GPU stack, and on a box with one
    # it only speaks after the caller has already waited. Same rule as
    # verify_stitch: structural checks first, machinery second.
    wrong = {n: p for n, p in expert_corpus_paths.items() if os.path.isdir(p)}
    if wrong:
        raise RuntimeError(
            f"train_router needs the JSONL CORPUS each expert trained on, but "
            f"got directories: {wrong}. That is almost always specialist_dirs "
            f"(the model checkpoints) passed where the corpus paths belong.")

    import torch

    out_dir = router_dir(config)
    if router_is_done(config):
        print(f"[skip] trained MoE (final) already present at {out_dir}")
        return out_dir

    print("\n⚡ Training router only...")

    # Load the MoE model
    if config.force or not os.path.exists(os.path.join(
            config.output_root, st.ARTIFACTS[st.STITCH], "config.json")):
        raise RuntimeError(
            "No MoE skeleton found. Run the stitch stage first.")

    # Load options
    load_kwargs = {"dtype": torch.bfloat16, "device_map": {"": 0},
                   "cache_dir": config.hf_home}

    # Router 4-bit escape hatch: load frozen experts in nf4, keep router fp32
    if os.environ.get("MSMOE_ROUTER_4BIT", "").lower() in ("1", "true", "yes"):
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["lm_head"],
        )
        print("   MSMOE_ROUTER_4BIT: frozen experts in nf4 (~28 GB), router stays fp32")

    # Try direct load (moe_load) — streams from mmap, much lower peak memory
    if os.environ.get("MSMOE_DIRECT_LOAD", "").lower() in ("1", "true", "yes"):
        import moe_load
        moe = moe_load.load_direct(
            f"{config.output_root}/" + st.ARTIFACTS[st.STITCH],
            model_cls=cfg_module.Qwen2MoeForCausalLM,
            device="cuda:0",
            dtype=torch.bfloat16,
        )
    else:
        from transformers import Qwen2MoeForCausalLM
        moe = Qwen2MoeForCausalLM.from_pretrained(
            f"{config.output_root}/" + st.ARTIFACTS[st.STITCH],
            **load_kwargs,
        )

    moe.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})

    # Optionally skip dense-layer backward (saves memory for dense-heavy configs)
    if os.environ.get("MSMOE_SKIP_DENSE_BACKWARD", "").lower() in ("1", "true", "yes"):
        print("   MSMOE_SKIP_DENSE_BACKWARD: not forcing input grads")
    else:
        moe.enable_input_require_grads()

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        f"{config.output_root}/" + st.ARTIFACTS[st.STITCH],
        cache_dir=config.hf_home)

    # Freeze all parameters EXCEPT router gates
    ROUTER_SUFFIX = ".mlp.gate.weight"
    trainable_count = 0
    for name, param in moe.named_parameters():
        is_router = name.endswith(ROUTER_SUFFIX)
        param.requires_grad = is_router
        if is_router:
            trainable_count += param.numel()

    if trainable_count == 0:
        raise RuntimeError(
            "No router parameters matched " + ROUTER_SUFFIX +
            " — the MoE layout changed. Check: " +
            str([n for n, _ in moe.named_parameters() if "gate" in n][:6]))

    print(f"   router-only training: {trainable_count:,} trainable params")

    # Build stratified data mix
    pools: Dict[str, List[str]] = {}
    for name in safe_names:
        path = expert_corpus_paths.get(name)
        # PREFER THE TRAIN SPLIT, SO HELD-OUT STAYS HELD OUT.
        #
        # The mix used to be drawn from the WHOLE corpus, held-out rows
        # included. eval's routing probe then excludes any held-out row it
        # finds in the mix - held out by construction, which is the right
        # instinct - so a big enough mix eats a source's entire held-out set
        # and that expert silently vanishes from the routing table.
        #
        # It happened: router_mix_total 1200 -> 4000 consumed every usable
        # python held-out row, the probe reported "column maximum for 2/2"
        # over three experts, and p quietly regressed from 0.037 to 0.250
        # because the test had narrowed without saying so.
        #
        # `.train` is written by _load_or_split at the same seed every time,
        # so when it exists it is exactly the complement of what eval will
        # hold out. Falling back to the full corpus keeps this working for
        # anyone who trains a router without ever running the gate or eval.
        if path:
            train_split = path + ".train"
            if os.path.exists(train_split) and os.path.getsize(train_split) > 0:
                path = train_split
        if not path or not os.path.exists(path):
            print(f"   WARNING: no data for expert {name!r} at {path}")
            continue
        with open(path, encoding="utf-8") as fh:
            pools[name] = [json.loads(line)["text"] for line in fh if line.strip()]

    if not pools:
        raise RuntimeError("no expert datasets found — nothing to train the router on")

    # Compute quotas
    quotas: Dict[str, int] = {}
    if "agentcore" in pools:
        quotas["agentcore"] = int(config.router_mix_total * config.agent_mix_fraction)

    code_names = [n for n in pools if n != "agentcore"]
    rest = config.router_mix_total - sum(quotas.values())
    for i, n in enumerate(code_names):
        quotas[n] = rest // len(code_names) + (1 if i < rest % len(code_names) else 0)

    # Handle starved sources — their shortfall goes to code experts
    take = {n: min(quotas[n], len(pools[n])) for n in pools}
    short = config.router_mix_total - sum(take.values())
    while short > 0:
        headroom = {n: len(pools[n]) - take[n] for n in pools
                    if n != "agentcore" and len(pools[n]) > take[n]}
        if not headroom:
            break
        per = max(1, short // len(headroom))
        for n in headroom:
            add = min(per, headroom[n], short)
            take[n] += add
            short -= add
            if short <= 0:
                break

    # Build mixed dataset
    mixed = []
    for n in pools:
        mixed.extend({"text": t, "src": n} for t in random.sample(pools[n], take[n]))
    random.shuffle(mixed)

    print(f"   router mix ({len(mixed)} rows, target {config.router_mix_total}):")
    for n in sorted(take, key=lambda k: -take[k]):
        pct = 100.0 * take[n] / max(len(mixed), 1)
        flag = "" if take[n] >= quotas.get(n, 0) else "   <-- SHORT of quota"
        print(f"      {n:12} {take[n]:>7}  ({pct:4.1f}%  pool {len(pools[n])}"
              f"  quota {quotas.get(n, 0)}){flag}")

    # Write mixed dataset
    mixed_path = f"{config.output_root}/mixed_all.jsonl"
    with open(mixed_path, "w", encoding="utf-8") as f:
        for s in mixed:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Load as dataset
    from datasets import load_dataset
    dataset = load_dataset("json", data_files=mixed_path, split="train")

    # Format
    def format_fn(ex):
        if ex["src"] == "agentcore":
            return ex["text"]
        lang = ex["src"]
        from .config import DISPLAY_LANG
        display = DISPLAY_LANG.get(lang, lang)
        prompt = _make_code_prompt(lang, display, config.code_prompt_unnamed_fraction)
        msgs = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": ex["text"]}]
        return tokenizer.apply_chat_template(msgs, tokenize=False) + tokenizer.eos_token

    dataset = dataset.map(
        lambda x: {"text": format_fn(x)},
        remove_columns=["src"],
    )

    # Tokenize
    tokenized = dataset.map(
        lambda x: tokenizer(x["text"], truncation=True,
                            max_length=config.max_seq_length),
        remove_columns=["text"],
    )

    # Custom trainer that preserves num_items_in_batch
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

    class MoETrainer(Trainer):
        """Custom trainer — forwards num_items_in_batch for correct loss scaling."""

        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None, **kwargs):
            if num_items_in_batch is not None:
                inputs = {**inputs, "num_items_in_batch": num_items_in_batch}
            outputs = model(**inputs, output_router_logits=True)
            return (outputs.loss, outputs) if return_outputs else outputs.loss

    trainer = MoETrainer(
        model=moe,
        args=TrainingArguments(
            output_dir=f"{config.output_root}/moe_router_ckpts",
            per_device_train_batch_size=config.router_batch,
            gradient_accumulation_steps=config.router_accum,
            gradient_checkpointing=True,
            num_train_epochs=config.router_epochs,
            learning_rate=config.lr_router,
            warmup_steps=max(10, round(0.05 * config.router_mix_total
                                       / (config.router_batch * config.router_accum))),
            logging_steps=10,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            report_to="none",
            save_strategy="no",
        ),
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Train
    print("   starting router training...")
    trainer.train()

    # Save
    moe.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"🎉 Router-trained MoE saved to {out_dir}")
    return out_dir


def _make_code_prompt(safe_name: str, display: str,
                      unnamed_fraction: float = 0.25,
                      rnd=None) -> str:
    """Same as finetune._make_code_prompt — kept here to avoid import cycle."""
    if rnd is None:
        import random
        rnd = random

    if rnd.random() < unnamed_fraction:
        return rnd.choice([
            "Write code:",
            "Implement this.",
            "Here is some code:",
        ])

    templates = [
        "Write {lang}:",
        "Write a {lang} script for this.",
        "In {lang}, implement the following.",
        "Give me some {lang} code.",
        "{lang}, please:",
        "Can you write this in {lang}?",
    ]
    return rnd.choice(templates).format(lang=display)
