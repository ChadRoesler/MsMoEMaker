# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors
#
# Headless in-process entry point for the vendored Heretic ablation core.
# Extracted and adapted from heretic's `main.py` `run()`: the interactive TUI,
# Hugging Face upload, benchmark, chat and reproducibility paths were removed.
# Everything the Optuna optimization loop needs to run unattended is here.

import math
import os
import random
import time
import warnings
from dataclasses import asdict
from os.path import commonprefix

import optuna
import torch
import torch.nn.functional as F
import transformers
from optuna import Trial, TrialPruned
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState

from .config import ExportStrategy, Settings
from .evaluator import Evaluator
from .model import AbliterationParameters, Model
from .system import empty_cache
from .utils import format_exception, get_trial_parameters, load_prompts, print


def _setup(settings: Settings) -> None:
    """Seed + silence, mirroring the top of heretic's run()."""
    if "PYTORCH_ALLOC_CONF" not in os.environ and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    if settings.seed is None:
        settings.seed = random.randint(0, 2**32 - 1)

    transformers.set_seed(settings.seed)
    torch.set_grad_enabled(False)
    torch._dynamo.config.cache_size_limit = 64
    transformers.logging.set_verbosity_error()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=ExperimentalWarning)


def _auto_batch_size(settings: Settings, model: Model, good_prompts) -> int:
    print("Determining optimal batch size...")
    batch_size = 1
    best_batch_size = -1
    best_performance = -1.0

    while batch_size <= settings.max_batch_size:
        prompts = good_prompts * math.ceil(batch_size / len(good_prompts))
        prompts = prompts[:batch_size]

        try:
            model.get_responses(prompts)  # warmup: build the computation graph
            start_time = time.perf_counter()
            responses = model.get_responses(prompts)
            end_time = time.perf_counter()
        except Exception as error:
            if batch_size == 1:
                raise
            print(f"Failed at batch size {batch_size}: {format_exception(error)}")
            break

        response_lengths = [len(model.tokenizer.encode(response)) for response in responses]
        performance = sum(response_lengths) / (end_time - start_time)
        print(f"  batch size {batch_size}: {performance:.0f} tokens/s")

        if performance > best_performance:
            best_batch_size = batch_size
            best_performance = performance

        batch_size *= 2

    print(f"Chosen batch size: {best_batch_size}")
    return best_batch_size


def _auto_response_prefix(settings: Settings, model: Model, good_prompts, bad_prompts) -> None:
    print("Checking for common response prefix...")
    prefix_check_prompts = good_prompts[:100] + bad_prompts[:100]
    responses = model.get_responses_batched(prefix_check_prompts)

    settings.response_prefix = commonprefix(responses).rstrip(" ")

    if settings.response_prefix:
        for cot_initializer, closed_cot_block in settings.chain_of_thought_skips:
            if settings.response_prefix.startswith(cot_initializer):
                settings.response_prefix = closed_cot_block
                responses = model.get_responses_batched(prefix_check_prompts)
                additional_prefix = commonprefix(responses).rstrip(" ")
                if additional_prefix:
                    settings.response_prefix += additional_prefix
                break


def run_abliteration(settings: Settings) -> str:
    """Abliterate `settings.model` and save the result to `settings.save_directory`.

    Headless: no prompts. Requires, on the passed `Settings`:
      - `model`           HF id or local dir
      - `save_directory`  where to write the decensored model
      - `export_strategy` `merge` | `adapter`
      - `checkpoint_action` `continue` | `restart` (or None -> fresh if no study)
      - `trial_index`     int, or None to take the first Pareto-front trial

    Returns the save directory path.
    """
    if settings.save_directory is None:
        raise ValueError("run_abliteration requires settings.save_directory")
    if settings.export_strategy is None:
        raise ValueError("run_abliteration requires settings.export_strategy (merge | adapter)")

    _setup(settings)

    # Study checkpoint: a JSONL journal next to the model slug, so a crashed run
    # resumes instead of restarting.
    os.makedirs(settings.study_checkpoint_dir, exist_ok=True)
    study_checkpoint_file = os.path.join(
        settings.study_checkpoint_dir,
        "".join([(c if (c.isalnum() or c in ["_", "-"]) else "--") for c in settings.model])
        + ".jsonl",
    )
    if os.path.exists(study_checkpoint_file) and settings.checkpoint_action == "restart":
        os.unlink(study_checkpoint_file)
    lock_obj = JournalFileOpenLock(study_checkpoint_file)
    backend = JournalFileBackend(study_checkpoint_file, lock_obj=lock_obj)
    storage = JournalStorage(backend)

    model = Model(settings)

    print(f"Loading good prompts from {settings.good_prompts.dataset}...")
    good_prompts = load_prompts(settings, settings.good_prompts)
    print(f"Loading bad prompts from {settings.bad_prompts.dataset}...")
    bad_prompts = load_prompts(settings, settings.bad_prompts)

    if settings.batch_size == 0:
        settings.batch_size = _auto_batch_size(settings, model, good_prompts)

    if settings.response_prefix is None:
        _auto_response_prefix(settings, model, good_prompts, bad_prompts)

    evaluator = Evaluator(settings, model)

    if not evaluator.get_objective_names():
        raise ValueError(
            "No optimization objectives configured: at least one scorer must set "
            'optimization to "minimize" or "maximize".'
        )

    print("Calculating per-layer residual directions...")
    good_means = model.get_residuals_mean(good_prompts)
    bad_means = model.get_residuals_mean(bad_prompts)
    residual_directions = F.normalize(bad_means - good_means, p=2, dim=1)

    if settings.orthogonalize_direction:
        # Projected abliteration: subtract only the component orthogonal to the
        # "good" direction (https://huggingface.co/blog/grimjim/projected-abliteration).
        good_directions = F.normalize(good_means, p=2, dim=1)
        projection_vector = torch.sum(residual_directions * good_directions, dim=1)
        residual_directions = (
            residual_directions - projection_vector.unsqueeze(1) * good_directions
        )
        residual_directions = F.normalize(residual_directions, p=2, dim=1)
        del good_directions, projection_vector

    del good_means, bad_means
    empty_cache()

    trial_index = 0

    def objective(trial: Trial) -> tuple[float, ...]:
        nonlocal trial_index
        trial_index += 1
        trial.set_user_attr("index", trial_index)

        direction_scope = trial.suggest_categorical(
            "direction_scope", ["global", "per layer"]
        )

        last_layer_index = len(model.get_layers()) - 1

        # Discrimination is usually strongest slightly past the midpoint of the
        # stack. Always sampled (even for "per layer") because multivariate TPE
        # does not support conditional parameters.
        direction_index = trial.suggest_float(
            "direction_index", 0.4 * last_layer_index, 0.9 * last_layer_index
        )

        if direction_scope == "per layer":
            direction_index = None

        parameters = {}
        for component in model.get_abliterable_components():
            # MLP gets a negative lower bound (clamped to 0) so the optimizer can
            # fully disable it: ablating the MLP often hurts more than it helps.
            max_weight_lower_bound = -0.25 if component == "mlp.down_proj" else 0.8
            max_weight = max(
                0.0,
                trial.suggest_float(
                    f"{component}.max_weight", max_weight_lower_bound, 1.5
                ),
            )
            max_weight_position = trial.suggest_float(
                f"{component}.max_weight_position",
                0.6 * last_layer_index,
                1.0 * last_layer_index,
            )
            # min_weight sampled as a fraction of max_weight (TPE has no
            # variable-range params), then scaled below.
            min_weight = trial.suggest_float(f"{component}.min_weight", 0.0, 1.0)
            min_weight_distance = trial.suggest_float(
                f"{component}.min_weight_distance",
                1.0,
                max(0.6 * last_layer_index, 1.0),
            )

            parameters[component] = AbliterationParameters(
                max_weight=max_weight,
                max_weight_position=max_weight_position,
                min_weight=(min_weight * max_weight),
                min_weight_distance=min_weight_distance,
            )

        trial.set_user_attr("direction_index", direction_index)
        trial.set_user_attr("parameters", {k: asdict(v) for k, v in parameters.items()})

        print(f"Running trial {trial_index} of {settings.n_trials}...")
        for name, value in get_trial_parameters(trial).items():
            print(f"  {name} = {value}")

        model.reset_model()
        model.abliterate(residual_directions, direction_index, parameters)

        scores = evaluator.get_scores()
        objective_values = evaluator.get_objective_values(scores)
        for name, score in scores:
            print(f"  {name}: {score.rich_display}")

        trial.set_user_attr("scores", evaluator.get_paired_score_records(scores))
        return objective_values

    def objective_wrapper(trial: Trial) -> tuple[float, ...]:
        try:
            return objective(trial)
        except KeyboardInterrupt:
            trial.study.stop()
            raise TrialPruned()

    objective_names = evaluator.get_objective_names()
    directions = evaluator.get_objective_directions()

    study = optuna.create_study(
        sampler=TPESampler(
            n_startup_trials=settings.n_startup_trials,
            n_ei_candidates=128,
            multivariate=True,
            seed=settings.seed,
        ),
        storage=storage,
        directions=directions,
        study_name="heretic",
        load_if_exists=True,
    )
    study.set_user_attr("settings", settings.model_dump_json())
    study.set_user_attr("finished", False)

    trial_index = len(study.trials)
    if trial_index > 0:
        print("Resuming existing study.")

    try:
        study.optimize(
            objective_wrapper,
            n_trials=settings.n_trials - len(study.trials),
        )
    except KeyboardInterrupt:
        pass

    if len(study.trials) == settings.n_trials:
        study.set_user_attr("finished", True)

    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed_trials:
        raise RuntimeError(
            "No completed trials: the study was interrupted before the first trial "
            "finished. Re-run with checkpoint_action='continue' to resume."
        )

    sorted_trials = sorted(
        study.best_trials,
        key=lambda trial: tuple(
            next(
                (score["score"]["value"] for score in trial.user_attrs["scores"] if score["name"] == name),
                None,
            )
            for name in objective_names
        ),
    )

    if settings.trial_index is not None:
        trial = sorted_trials[settings.trial_index]
    else:
        trial = sorted_trials[0]

    print(f"Restoring model from trial {trial.user_attrs['index']}...")
    model.reset_model()
    model.abliterate(
        residual_directions,
        trial.user_attrs["direction_index"],
        {
            k: AbliterationParameters(**v)
            for k, v in trial.user_attrs["parameters"].items()
        },
    )

    save_directory = settings.save_directory
    if settings.export_strategy == ExportStrategy.ADAPTER:
        print("Saving LoRA adapter...")
        model.model.save_pretrained(save_directory, max_shard_size=settings.max_shard_size)
    else:
        print("Saving merged model...")
        merged_model = model.get_merged_model()
        merged_model.save_pretrained(save_directory, max_shard_size=settings.max_shard_size)
        del merged_model
        empty_cache()
        model.tokenizer.save_pretrained(save_directory)

    print(f"Model saved to {save_directory}.")
    return save_directory
