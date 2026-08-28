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
import time
from typing import Optional

from . import config as cfg_module
from . import stages as st


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


def specialist_is_done(config, safe_name: str) -> bool:
    """Has this expert already been trained?
# ONE PREDICATE, TWO CALLERS. The skip condition lives here and nowhere else,
# because the stage function and the orchestrator both need the answer and
# deriving it twice is how they drift. (Runner and build_config each worked out
# the run directory independently and disagreed - the manifest went to one
# folder and every artifact to another. Same trap, so: one function.)
    """
    if config.force:
        return False
    return os.path.exists(f"{specialist_dir(config, safe_name)}/config.json")


def fine_tune_specialist(config, safe_name: str, data_path: str,
                         expert_display: Optional[str] = None) -> str:
    """LoRA fine-tune one specialist expert.

    Resumes from checkpoint if a previous run was interrupted.  Saves the
    merged specialist (dense, not quantised) to OUTPUT_ROOT/specialist_{safe}.

    Returns the output directory path.
    """
    from . import config as cfg_module
    import torch

    out_dir = specialist_dir(config, safe_name)
    if specialist_is_done(config, safe_name):
        print(f"Specialist {safe_name} already trained at {out_dir}, skipping.")
        return out_dir

    print(f"\nFine-tuning {safe_name}...")

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
            print("[env] unsloth requested but not installed — falling back to plain")
            want_unsloth = False

    if want_unsloth and not unsloth_available:
        raise RuntimeError(
            "MSMOE_UNSLOTH=1 but unsloth is not installed. "
            "Either pip install unsloth or unset MSMOE_UNSLOTH.")

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
        from .config import DISPLAY_LANG
        display = DISPLAY_LANG.get(safe_name, safe_name)
        prompt = _make_code_prompt(safe_name, display, config.code_prompt_unnamed_fraction)
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

    # Build trainer
    tmp_dir = f"{config.output_root}/tmp_{safe_name}"

    train_kwargs = dict(
        per_device_train_batch_size=config.per_device_batch,
        gradient_accumulation_steps=config.grad_accum,
        warmup_steps=config.warmup_steps,
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
    have_ckpt = os.path.isdir(tmp_dir) and any(
        d.startswith("checkpoint-") for d in os.listdir(tmp_dir))
    if have_ckpt:
        print(f"   resuming {safe_name} from checkpoint in {tmp_dir}")
        trainer.train(resume_from_checkpoint=have_ckpt)
    else:
        trainer.train()

    # Save merged specialist (dense, NOT quantised)
    if config.load_in_4bit:
        if not hasattr(model, "save_pretrained_merged"):
            raise RuntimeError(
                "LOAD_IN_4BIT=True needs unsloth's save_pretrained_merged. "
                "Set LOAD_IN_4BIT=False and retrain.")
        model.save_pretrained_merged(out_dir, tokenizer, save_method="merged_16bit")
    else:
        merged = model.merge_and_unload()
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
