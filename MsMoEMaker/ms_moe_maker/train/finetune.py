"""LoRA specialist training — fine-tune one expert on its corpus.

Supports two backends:
  - Plain transformers + PEFT (fastest on GB10 / A-series, 5.5x unsloth)
  - Unsloth (optional, for boxes where its kernels are actually fast)

Uses TRL's SFTTrainer with 'wrapped' packing (no padding-free).  Saves
checkpoint-cadenced intermediates so a crash at expert 4 does not lose
experts 1–3.

The specialist is saved as REAL DENSE WEIGHTS (not bnb-packed) so the
stitcher can copy tensors into the MoE skeleton.
"""
from __future__ import annotations

import os
import json
import hashlib
import random
import shutil
import time
from typing import Optional

from ..config import pipeline as cfg_module
from ..run import stages as st


def make_heartbeat_callback(print_interval: int = 100,
                            checkpoint_interval: float = 300):
    """A progress callback the Trainer will actually accept.

    THIS WAS A BARE CLASS, and that is why the first real 0.5B build died at
    stage 3 with:

        AttributeError: 'HeartbeatCallback' object has no attribute 'on_init_end'

    transformers' CallbackHandler does `getattr(callback, event)(...)` for
    EVERY lifecycle hook - on_init_end, on_train_begin, on_step_begin,
    on_optimizer_step and the rest - not just the ones you happened to write.
    Duck-typing loses against a framework that introspects the full interface;
    TrainerCallback exists precisely to supply the no-op defaults, and
    subclassing it is not optional.

    It could not simply subclass at module level, though, and that constraint
    is real rather than an oversight: `from transformers import TrainerCallback`
    at import time would put torch behind `ms-moe-maker validate` and break the
    laptop promise. Hence a FACTORY - a module-level function that imports
    lazily and builds the subclass on the first call that needs it. The class
    keeps the full interface, the module keeps its stdlib-only import.

    Worth noting the failure shape: everything before this worked. The corpus
    collected, the base cached, 1635 docs tokenized and packed - and then the
    trainer refused to be constructed. Anything that fails at CONSTRUCTION
    after a long setup is expensive in exactly this way, which is an argument
    for building the trainer earlier or checking the callback shape at
    preflight.
    """
    from transformers import TrainerCallback

    class HeartbeatCallback(TrainerCallback):
        """Prints progress every N steps / minutes."""

        def __init__(self):
            super().__init__()
            self.print_interval = print_interval
            self.checkpoint_interval = checkpoint_interval
            self.last_print = time.time()
            self.step_count = 0

        def on_step_end(self, args, state, control, **kwargs):
            self.step_count += 1
            if self.step_count % self.print_interval == 0:
                print(f"   Still training... step {self.step_count}")
            if time.time() - self.last_print > self.checkpoint_interval:
                loss_val = "?"
                if state.log_history:
                    loss_val = state.log_history[-1].get("loss", "?")
                print(f"   Training checkpoint: step {self.step_count}, "
                      f"loss ~ {loss_val}")
                self.last_print = time.time()
            return control

    return HeartbeatCallback()


def _ensure_cached(repo_id: str, hf_home: str, retries: int = 4):
    """Fetch model weights with the plain hub client before any other import."""
    if os.path.isdir(repo_id):
        return

    from huggingface_hub import snapshot_download
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    for attempt in range(1, retries + 1):
        try:
            snapshot_download(repo_id, cache_dir=hf_home)
            print(f"   base weights cached: {repo_id}")
            return
        except Exception as exc:
            wait = 15 * attempt
            print(f"   fetch attempt {attempt}/{retries} failed "
                  f"({type(exc).__name__}: {str(exc)[:120]})")
            if attempt == retries:
                raise RuntimeError(
                    f"could not fetch {repo_id} after {retries} attempts. "
                    f"Pull it by hand and re-run:\n"
                    f"    HF_HUB_DISABLE_XET=1 hf download {repo_id}"
                ) from exc
            print(f"   retrying in {wait}s...")
            time.sleep(wait)


def specialist_dir(config, safe_name: str) -> str:
    """Where this expert's merged specialist lives."""
    return f"{config.output_root}/" + st.FINETUNE_ARTIFACT.format(expert=safe_name)


def specialist_is_done(config, safe_name: str, retrain: bool = False) -> bool:
    """Has this expert already been trained?

# `retrain` IS THE PER-EXPERT --force, and it is what `--only <expert>` pulls.
# The flags used to be all-or-nothing: --force redid all N experts, and the
# only way to redo ONE was to delete its directory by hand - which retrained
# it and then had the stitch skip, because the stitch only compared names.
# The caller says which experts it named; the predicate stays the single
# source of the skip decision.
# ONE PREDICATE, TWO CALLERS. The skip condition lives here and nowhere else,
# because the stage function and the orchestrator both need the answer and
# deriving it twice is how they drift. (Runner and build_config each worked out
# the run directory independently and disagreed - the manifest went to one
# folder and every artifact to another. Same trap, so: one function.)

# NOT PRESENCE-ONLY. `merged.save_pretrained` writes config.json FIRST, then
# multi-GB weight shards, and the tokenizer in a separate call after that - a
# kill / OOM / disk-full anywhere in that window used to leave a config.json
# that read as "already trained", so resume SKIPPED the expert and the stitch
# stitched a directory with no weights in it. All three markers must exist.
    """
    if config.force or retrain:
        return False
    d = specialist_dir(config, safe_name)
    if not os.path.exists(os.path.join(d, "config.json")):
        return False
    has_weights = (os.path.exists(os.path.join(d, "model.safetensors"))
                   or os.path.exists(os.path.join(d, "pytorch_model.bin")))
    has_tokenizer = os.path.exists(os.path.join(d, "tokenizer_config.json"))
    return has_weights and has_tokenizer


def expert_seed(seed: int, safe_name: str) -> int:
    """A stable per-expert seed derived from the build's `seed`.

    Two reasons it is not just `config.seed`:

    Seeding every expert off the same number hands python and rust the
    IDENTICAL prompt sequence - correlated rather than random, which is a
    different kind of wrong from unseeded and harder to notice. Mixing the
    expert name in keeps the streams independent while keeping the whole set
    reproducible from one number.

    And it is sha256, not the builtin hash(): hash() of a str is salted per
    interpreter (PYTHONHASHSEED), so it would answer differently in every
    process - which defeats the entire point. `seed` is inside the build
    fingerprint (config.pipeline.build_fingerprint keeps every field it does
    not explicitly exclude), so build_id has always CLAIMED two runs trained
    the same way. Until this was wired, that claim was only ever about what
    the config asked for: nothing in train/ called set_seed, and the prompt
    attached to every row came from the unseeded module-level `random`.
    """
    blob = f"{int(seed)}:{safe_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:4], "big") % (2 ** 31 - 1)


def _checkpoint_dirs(tmp_dir: str) -> list:
    """`checkpoint-N` dirs under tmp_dir, lowest step first.

    Sorted by the STEP NUMBER, not by name: the old resume check was a bare
    `d.startswith("checkpoint-")`, and anything that sorts checkpoint dirs as
    strings puts checkpoint-9 after checkpoint-100. Anything that is not
    `checkpoint-<int>` is not a checkpoint and is ignored rather than counted.
    """
    if not os.path.isdir(tmp_dir):
        return []
    found = []
    for d in os.listdir(tmp_dir):
        if not d.startswith("checkpoint-"):
            continue
        if not os.path.isdir(os.path.join(tmp_dir, d)):
            continue
        try:
            step = int(d.split("-", 1)[1])
        except ValueError:
            continue
        found.append((step, d))
    return [d for _, d in sorted(found)]


def latest_checkpoint(tmp_dir: str) -> Optional[str]:
    """Full path of the highest-step checkpoint under tmp_dir, or None."""
    dirs = _checkpoint_dirs(tmp_dir)
    return os.path.join(tmp_dir, dirs[-1]) if dirs else None


def checkpoint_global_step(ckpt_dir: str) -> int:
    """The optimiser step a checkpoint would resume AT. 0 if unreadable.

    0 is the honest answer for a checkpoint transformers could not resume from
    either - it reads the same trainer_state.json - and it keeps the zero-step
    guard below meaningful instead of raising a second, less useful error.
    """
    try:
        with open(os.path.join(ckpt_dir, "trainer_state.json"),
                  encoding="utf-8") as fh:
            return int(json.load(fh).get("global_step", 0) or 0)
    except (OSError, ValueError, TypeError):
        return 0


def clear_checkpoints(tmp_dir: str) -> list:
    """Delete every `checkpoint-N` under tmp_dir. Returns the names removed.

    Only the checkpoint dirs, not the whole tmp dir: the trainer owns this
    folder but a human may well have dropped a log or a note in it, and the
    thing that has to go is specifically the state a resume would latch onto.
    """
    removed = _checkpoint_dirs(tmp_dir)
    for d in removed:
        shutil.rmtree(os.path.join(tmp_dir, d), ignore_errors=True)
    return removed


def _refuse_adapter_base(model, base_label: str) -> None:
    """An adapter checkpoint is NOT a valid training base. Refuse, loudly, BEFORE the expensive part.

    transformers/peft auto-load an adapter whenever a model directory contains
    adapter_config.json - so pointing config.base at an adapter export trains
    from weights with the delta applied TWICE (the export also holds the
    merged model), and our LoRA then nests inside the auto-loaded adapter. The
    first gauntlet run hit exactly that: training finished, the merged model
    still carried peft_config, and save_pretrained took the adapter path
    (get_adapter_state_dict -> active_adapters) and died with an upstream
    UnboundLocalError. Like the 4-bit and unsloth guards below, this fails
    before the bill is paid rather than after.
    """
    if getattr(model, "peft_config", None):
        raise RuntimeError(
            f"config.base {base_label!r} is an ADAPTER checkpoint: it carries "
            "peft_config, so loading it auto-applied the adapter delta on top "
            "of weights that already contain it (double-applied), and LoRA "
            "would nest inside the auto-loaded adapter. Point config.base at "
            "the MERGED checkpoint instead - an adapter export keeps its "
            "merged model at the directory root and the delta in the "
            "adapter/ subdir.")


def _strip_peft_residue(model, label: str) -> bool:
    """Drop adapter residue from a merged specialist before the dense save.

    Returns True if something was stripped. If the residue cannot be removed
    (a class-level property, say), refusing is the only honest option:
    save_pretrained would route through get_adapter_state_dict, which either
    writes ADAPTER files where dense weights belong or crashes first - on the
    transformers version whose active_adapters references an unbound local.
    """
    if getattr(model, "peft_config", None) is None:
        return False
    try:
        del model.peft_config
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(
            f"{label}: the merged specialist still carries peft adapter state "
            f"that cannot be removed, and save_pretrained would treat it as an "
            f"adapter checkpoint; refusing to write adapter-flavoured weights "
            f"where a dense specialist belongs."
        ) from exc
    return True


def fine_tune_specialist(config, safe_name: str, data_path: str,
                         expert_display: Optional[str] = None,
                         retrain: bool = False) -> str:
    """LoRA fine-tune one specialist expert.

    Resumes from checkpoint if a previous run was interrupted.  Saves the
    merged specialist (dense, not quantised) to OUTPUT_ROOT/specialist_{safe}.

    Returns the output directory path.
    """
    from ..config import pipeline as cfg_module
    import torch

    out_dir = specialist_dir(config, safe_name)
    if specialist_is_done(config, safe_name, retrain):
        print(f"Specialist {safe_name} already trained at {out_dir}, skipping.")
        return out_dir

    print(f"\nFine-tuning {safe_name}...")

    # SEED FIRST, BEFORE ANYTHING DRAWS.
    #
    # `config.seed` had exactly one consumer - the router init in
    # moe/stitch.py - so nothing in the training path was seeded at all. That
    # matters most for the prompt: _make_code_prompt drew from the unseeded
    # module-level `random` inside dataset.map, which happens BELOW here and
    # well before SFTTrainer is constructed and calls set_seed(args.seed). So
    # every row of every expert got a prompt from a stream that differed run
    # to run, and re-running ONE expert with --force gave it a different
    # prompt distribution from its neighbours - the one thing this pipeline
    # spends whole stages trying to keep comparable.
    #
    # Placed above the model load on purpose: LoRA init draws too, and a seed
    # set after it seeds nothing that matters.
    from transformers import set_seed
    seed = expert_seed(config.seed, safe_name)
    set_seed(seed)
    prompt_rnd = random.Random(seed)
    print(f"   seed: build {config.seed} -> {safe_name} stream {seed}")

    # Check unsloth availability
    want_unsloth = config.use_unsloth
    unsloth_available = False
    unsloth = None

    if want_unsloth:
        try:
            import unsloth  # noqa: F401  (patches on import)
            unsloth_available = True
            unsloth = unsloth
            print("[env] unsloth IMPORTED")
        except ImportError:
            # FALL BACK, AND SAY WHAT IT COSTS. A plain fine-tune is a real
            # result, so this is not a refusal - but the one-line print it
            # replaces understated the situation twice over.
            #
            # `config.optim` was resolved to "adamw_8bit" from use_unsloth
            # back in build_config, and the line that hands the trainer
            # `optim=config.optim` does not care which path we took - so the
            # plain path runs with the 8-bit optimiser unsloth was going to
            # supply, and needs bitsandbytes to honour it.
            #
            # And use_unsloth is in the build fingerprint, so the manifest
            # then describes a run that did not happen. Preflight warns about
            # all of this before the GPU is booked (_check_trainer); this is
            # the same sentence for anyone who got past it.
            print("[env] WARNING: unsloth was requested (MSMOE_UNSLOTH) and "
                  "is not installed - training on the PLAIN path.")
            print(f"[env]          optim stays {config.optim!r}, resolved "
                  f"from use_unsloth, and the plain trainer needs "
                  f"bitsandbytes to honour it.")
            print("[env]          the manifest records use_unsloth=true for "
                  "a run that did not use it. Install unsloth, or unset "
                  "MSMOE_UNSLOTH so the record matches the run.")
            want_unsloth = False

    # NO `if want_unsloth and not unsloth_available: raise` HERE. There was
    # one, and it could never fire: the except above sets want_unsloth to
    # False, so the condition was unreachable and the RuntimeError it carried
    # - "MSMOE_UNSLOTH=1 but unsloth is not installed" - was dead code
    # standing in for a guard nobody had. The real behaviour was, and remains,
    # a fallback; it is a loud one now, said here and at preflight.


    # 4-BIT TRAINING FAILS FAST, BEFORE THE EXPENSIVE PART. The merged save
    # needs unsloth's save_pretrained_merged; the plain path used to discover
    # that AFTER trainer.train() had already finished - i.e. after the whole
    # bill had been paid. Refuse up front, with the recipe-level remedy.
    if config.load_in_4bit and not unsloth_available:
        raise RuntimeError(
            "runtime.load_in_4bit=true needs unsloth (its "
            "save_pretrained_merged produces the mergeable specialist). "
            "Refusing BEFORE training: set runtime.load_in_4bit: false in the "
            "recipe, or install unsloth and set MSMOE_UNSLOTH=1.")

    # Lazy import TRL
    try:
        from trl import SFTTrainer, SFTConfig
    except ImportError:
        from transformers import Trainer, TrainingArguments
        SFTConfig = None

    _ensure_cached(config.base, config.hf_home)

    # Load model
    if unsloth_available and want_unsloth:
        model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
            model_name=config.base,
            max_seq_length=config.max_seq_length,
            dtype=None,
            load_in_4bit=config.load_in_4bit,
            device_map={"": 0},
            cache_dir=config.hf_home,
        )
        model = unsloth.FastLanguageModel.get_peft_model(
            model,
            r=config.lora_r,
            target_modules=config.target_modules,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )
    else:
        # Plain transformers + PEFT
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        quant = None
        if config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        model = AutoModelForCausalLM.from_pretrained(
            config.base,
            dtype=torch.bfloat16,
            device_map={"": 0},
            quantization_config=quant,
            attn_implementation=config.attn_impl,
            cache_dir=config.hf_home,
        )
        _refuse_adapter_base(model, config.base)

        model = get_peft_model(model, LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=config.target_modules,
        ))

        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

        tokenizer = AutoTokenizer.from_pretrained(
            config.base, cache_dir=config.hf_home)

    # Format function
    def format_fn(ex):
        if (safe_name == config.tools_expert_name
                or safe_name in config.reasoning_experts
                or safe_name in config.synth_experts):
            return ex["text"]
        lang = expert_display or safe_name
        from ..config.pipeline import DISPLAY_LANG
        display = DISPLAY_LANG.get(safe_name, safe_name)
        # rnd= is why _make_code_prompt has that parameter: without it the
        # draw is the process-global `random` and the prompts are unrepeatable.
        prompt = _make_code_prompt(safe_name, display,
                                   config.code_prompt_unnamed_fraction,
                                   rnd=prompt_rnd)
        msgs = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": ex["text"]}]
        return tokenizer.apply_chat_template(msgs, tokenize=False) + tokenizer.eos_token

    # Load dataset
    from datasets import load_dataset
    dataset = load_dataset("json", data_files=data_path, split="train")

    # Cap at the collection ceiling
    cap = (config.num_agent_samples
           if (safe_name == config.tools_expert_name
               or safe_name in config.reasoning_experts
               or safe_name in config.synth_experts)
           else config.num_code_samples)
    if len(dataset) > cap:
        print(f"   dataset has {len(dataset)} rows, capping to {cap}")
        dataset = dataset.select(range(cap))

    # Format
    dataset = dataset.map(
        lambda x: {"text": format_fn(x)},
        remove_columns=["text"],
    )

    # Token-budget trim
    dataset = _trim_to_token_budget(
        dataset, tokenizer, config.expert_token_budget, safe_name,
        config.max_seq_length, config.per_device_batch, config.grad_accum,
        config.chars_per_token_est,
    )

    # WARMUP FROM THE STEPS THE RUN WILL ACTUALLY TAKE, not target_steps.
    # The trim above is the real schedule: a short corpus means fewer steps,
    # and target_steps-based warmup could spend 25-100% of the run ramping -
    # at the README's small-run settings the entire run was warmup and the LR
    # never reached lr_lora. Cap at half the schedule so warmup can never BE
    # the schedule.
    planned_steps = max(
        1, len(dataset) // (config.per_device_batch * config.grad_accum))
    warmup_steps = min(
        max(config.warmup_floor, round(config.warmup_ratio * planned_steps)),
        planned_steps // 2)

    # Build trainer
    tmp_dir = f"{config.output_root}/tmp_{safe_name}"

    # --force MEANS RETRAIN. A LEFTOVER CHECKPOINT QUIETLY MADE IT MEAN RESUME.
    #
    # specialist_is_done() short-circuits on config.force, so --force gets you
    # past the "already trained, skipping" gate - but nothing ever cleared
    # tmp_{expert}, and the resume branch below only asked "is there a
    # checkpoint-* dir in there".
    #
    # The run that costs you a build: run 1 dies at step 300 of 1200. You lower
    # target_steps to 150 - which is exactly what cli/build.py tells you to do
    # when drift refuses - and re-run with --force. The shorter schedule plans
    # ~150 steps, the inherited trainer_state.json says global_step=200, the
    # Trainer works out epochs_trained >= 1 and skips the epoch loop body
    # entirely. train() returns having taken ZERO optimiser steps,
    # merge_and_unload() saves the base weights with an untouched LoRA, and the
    # builder records "finetune.X -> done, saved". An untrained expert,
    # reported trained, then stitched into the MoE.
    # `retrain` gets the same treatment as --force and for the same reason:
    # --only <expert> means retrain that expert, and a leftover checkpoint
    # would quietly turn it back into a resume - the exact trap described
    # above, just entered through the other flag.
    if config.force or retrain:
        gone = clear_checkpoints(tmp_dir)
        if gone:
            _why = "--force" if config.force else f"--only {safe_name}"
            print(f"   {_why}: discarded {len(gone)} stale checkpoint(s) in "
                  f"{tmp_dir} ({', '.join(gone)}). Forcing means retraining, "
                  f"not resuming.")

    train_kwargs = dict(
        # THE TRAINER SEEDS ITSELF, FROM ITS OWN DEFAULT, IF YOU LET IT.
        # TrainingArguments.seed defaults to 42 and the Trainer calls
        # set_seed(args.seed) on construction - so a recipe that asked for any
        # other seed would have had it overwritten somewhere between the model
        # load and the first batch. Hand it the same stream we used above.
        seed=seed,
        per_device_train_batch_size=config.per_device_batch,
        gradient_accumulation_steps=config.grad_accum,
        warmup_steps=warmup_steps,
        num_train_epochs=1,
        learning_rate=config.lr_lora,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim=config.optim,
        output_dir=tmp_dir,
        save_strategy="steps",
        save_steps=config.specialist_save_steps,
        save_total_limit=2,
        report_to="none",
    )

    if SFTConfig is not None:
        import inspect
        cfg_params = set(inspect.signature(SFTConfig.__init__).parameters)
        seq_key = "max_length" if "max_length" in cfg_params else "max_seq_length"
        cfg_extra = {
            "dataset_text_field": "text",
            "packing": True,
            seq_key: config.max_seq_length,
        }
        if "padding_free" in cfg_params:
            cfg_extra["padding_free"] = False
        if "packing_strategy" in cfg_params:
            cfg_extra["packing_strategy"] = config.packing_strategy
        args = SFTConfig(**train_kwargs, **cfg_extra)
        sft_extra = {}
    else:
        args = TrainingArguments(**train_kwargs)  # type: ignore[name-defined]
        sft_extra = dict(
            dataset_text_field="text",
            packing=True,
            max_seq_length=config.max_seq_length,
        )

    # Tokenizer kwarg detection
    sft_params = set(inspect.signature(SFTTrainer.__init__).parameters)
    tok_key = "processing_class" if "processing_class" in sft_params else "tokenizer"

    print(f"   trl api: args={type(args).__name__}, tokenizer kwarg={tok_key}")

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=args,
        callbacks=[make_heartbeat_callback()],
        **{tok_key: tokenizer},
        **sft_extra,
    )

    # Resume from checkpoint
    ckpt = latest_checkpoint(tmp_dir)
    resumed_at = checkpoint_global_step(ckpt) if ckpt else 0
    if ckpt:
        print(f"   resuming {safe_name} from {os.path.basename(ckpt)} "
              f"(global_step {resumed_at})")
        trainer.train(resume_from_checkpoint=ckpt)
    else:
        trainer.train()

    # DID IT ACTUALLY TRAIN? ASK AFTERWARDS - DO NOT PREDICT.
    #
    # The stale-checkpoint story above is the common way to end up here, and
    # --force now prevents that one. This is the general case: ANY resume whose
    # global_step is already at or past the end of the current schedule skips
    # the epoch loop and returns quietly, and the next lines would merge and
    # save an adapter that never saw a gradient.
    #
    # Checked after the fact rather than by computing the planned step count
    # up front, because "planned steps" means re-deriving the Trainer's own
    # len(dataloader) // grad_accum arithmetic and staying bug-compatible with
    # it forever, while state.global_step is the actual answer and free. Loud
    # beats clever: refuse to save rather than save something untrained.
    if trainer.state.global_step <= resumed_at:
        raise RuntimeError(
            f"{safe_name}: training took ZERO optimiser steps "
            f"(global_step {resumed_at} -> {trainer.state.global_step}). "
            f"A checkpoint at step {resumed_at} is at or past the end of "
            f"the schedule this run planned, so there was nothing left to do "
            f"and the adapter is UNTRAINED - saving it would report a specialist that never "
            f"learned anything. Delete {tmp_dir} and re-run, or raise the step "
            f"budget above {resumed_at}.")

    # Save merged specialist (dense, NOT quantised)
    # `merged` is bound only in the else branch; initialising it here means
    # the 4-bit path can reach the shared `del` below without a NameError -
    # which used to mark a run FAILED with the specialist already on disk.
    merged = None
    if config.load_in_4bit:
        if not hasattr(model, "save_pretrained_merged"):
            raise RuntimeError(
                "runtime.load_in_4bit=true but this model has no "
                "save_pretrained_merged (unsloth was not active). Set "
                "runtime.load_in_4bit: false and retrain.")
        model.save_pretrained_merged(out_dir, tokenizer, save_method="merged_16bit")
    else:
        merged = model.merge_and_unload()
        # DENSE SAVE, NEVER THE ADAPTER PATH. Some transformers/peft pairs
        # leave peft_config on the unloaded model; save_pretrained then
        # routes through get_adapter_state_dict -> active_adapters(), which
        # is exactly where the first gauntlet build died (an upstream
        # UnboundLocalError). The specialist on disk must be plain dense
        # weights, so drop any residue before the save.
        if _strip_peft_residue(merged, safe_name):
            print(f"   {safe_name}: stripped peft residue from the merged "
                  f"specialist before the dense save")
        merged.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)

    # Verify: saved weights are truly dense
    saved_cfg = json.load(open(f"{out_dir}/config.json"))
    if saved_cfg.get("quantization_config") is not None:
        raise RuntimeError(
            f"{out_dir} saved STILL QUANTISED. Delete and retrain with "
            f"load_in_4bit=False.")

    print(f"Dense specialist saved to {out_dir}")

    # Cleanup
    del model, merged, trainer
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    return out_dir


def _make_code_prompt(safe_name: str, display: str,
                      unnamed_fraction: float = 0.25,
                      rnd=None) -> str:
    """One user turn for a code sample.  Names the language most of the time."""
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


def _trim_to_token_budget(dataset, tokenizer, budget: int,
                          label: str, max_seq: int,
                          batch_size: int, grad_accum: int,
                          chars_per_token: float = 3.2):
    """Cut dataset to a TOKEN budget.  The unit that actually equalises experts."""
    print(f"   measuring {label} tokens")
    counted = dataset.map(
        lambda b: {"_ntok": [len(i) for i in
                             tokenizer(b["text"], add_special_tokens=False)["input_ids"]]},
        batched=True,
        batch_size=256,
        desc=f"   measuring {label} tokens",
    )
    lens = counted["_ntok"]

    total, keep = 0, 0
    for n in lens:
        if total >= budget:
            break
        total += n
        keep += 1

    corpus = sum(lens)
    per_step = max_seq * batch_size * grad_accum
    steps = total // per_step
    mean_tok = corpus / max(len(lens), 1)

    print(f"   token budget {label}: {total/1e6:.2f}M of {budget/1e6:.2f}M "
          f"from {keep}/{len(lens)} docs (~{mean_tok:.0f} tok/doc) "
          f"-> ~{steps} steps")

    if total < budget:
        short_docs = (budget - corpus) / max(mean_tok, 1)
        print(f"   *** {label} is SHORT of budget: corpus holds only "
              f"{corpus/1e6:.2f}M. Trains on {steps}-step schedule vs "
              f"{budget // per_step}. Top up with ~{short_docs:.0f} more docs, "
              f"or lower MSMOE_TARGET_STEPS.")

    return dataset.select(range(keep))
