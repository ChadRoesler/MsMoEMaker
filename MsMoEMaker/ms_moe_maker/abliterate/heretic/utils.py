# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors
#
# Vendored from p-e-w/heretic and trimmed to the ablation core. Dropped the
# reproducibility/upload helpers, the interactive `ask_if_unset`, memory/format
# helpers, and their deps (questionary, psutil, tomli_w, importlib.metadata).

import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import huggingface_hub
from datasets import DatasetDict, ReadInstruction, load_dataset, load_from_disk
from datasets.config import DATASET_STATE_JSON_FILENAME
from datasets.download.download_manager import DownloadMode
from datasets.utils.info_utils import VerificationMode
from huggingface_hub.utils import validate_repo_id
from optuna import Trial
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from rich.console import Console

from .config import DatasetSpecification, Settings

T = TypeVar("T")

print = Console(highlight=False).print


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge two dicts.

    Values from `override` take precedence. Nested dicts are merged recursively.
    """
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = value
    return merged


def parse_study_direction(optimization: str) -> StudyDirection:
    """
    Converts the optimization value stored as a `str` to the
    `StudyDirection` object required by Optuna.
    """
    if optimization == "none":
        return StudyDirection.NOT_SET
    return StudyDirection[optimization.upper()]


def format_exception(error: Exception) -> str:
    # Walk causal chain to find a non-empty message.
    current = error
    while current is not None:
        message = str(current).strip()
        if message:
            return message
        current = current.__cause__ or current.__context__

    # If there is no message in the entire causal chain, fall back to the complete traceback.
    return traceback.format_exc().strip()


def is_hf_path(path: str) -> bool:
    """Checks whether a path likely refers to a Hugging Face repository."""

    # Match Transformers: Existing local paths take precedence over Hub lookup,
    # even if the path string is also a valid repository ID.
    if Path(path).exists():
        return False

    validate_repo_id(path)
    return True


@dataclass
class Prompt:
    system: str
    user: str


def get_split_slice(split_str: str, length: int) -> tuple[int, int]:
    """Resolves a split specification into absolute (start, end) indices."""

    # The split name is the part before the slice, e.g. "train" in "train[:400]".
    split_name = split_str.split("[")[0]

    # Associate the split with its number of examples (lines).
    name_to_length = {split_name: length}

    # Convert the instructions to absolute indices and select the first one.
    absolute_instruction = ReadInstruction.from_spec(split_str).to_absolute(
        name_to_length
    )[0]

    return absolute_instruction.from_, absolute_instruction.to


def load_prompts(
    settings: Settings,
    specification: DatasetSpecification,
) -> list[Prompt]:
    path = specification.dataset
    split_str = specification.split

    if os.path.isfile(path):
        # Plain text file with one prompt per line. Empty lines are ignored.
        with open(path, encoding="utf-8") as file:
            prompts = [line.strip() for line in file if line.strip()]

        # The split is optional for text files. When given, it selects a subset
        # of the lines using slice notation (e.g. "[:400]"). A synthetic split
        # name is prepended because ReadInstruction expects a named split.
        if split_str is not None:
            start, end = get_split_slice(f"_{split_str}", len(prompts))
            prompts = prompts[start:end]
    else:
        # All dataset sources require an explicit split and column.
        if split_str is None:
            raise ValueError(f'The "split" field is required for datasets: {path}')

        if specification.column is None:
            raise ValueError(f'The "column" field is required for datasets: {path}')

        if is_hf_path(path):
            # Pin to the latest commit if not already set, so the exact dataset
            # version is recorded for reproducibility.
            if specification.commit is None:
                try:
                    specification.commit = huggingface_hub.dataset_info(path).sha
                except Exception as error:
                    # Fetching the commit hash requires internet access, but the
                    # dataset itself may be fully cached locally. Proceed without
                    # pinning; an unpinned dataset disables the reproducibility
                    # offer during upload.
                    print(
                        f"[yellow]Warning: Could not fetch the latest commit hash for dataset [bold]{path}[/] ({error}). "
                        "The dataset version will not be pinned.[/]"
                    )
            dataset = load_dataset(
                path,
                revision=specification.commit,
                split=split_str,
            )
        elif Path(path, DATASET_STATE_JSON_FILENAME).exists():
            # Dataset saved with datasets.save_to_disk; needs special handling.
            # Path should be the subdirectory for a particular split.
            dataset = load_from_disk(path)
            assert not isinstance(dataset, DatasetDict), (
                "Loading dataset dicts is not supported"
            )
            # Parse the split instructions and apply them.
            start, end = get_split_slice(split_str, len(dataset))
            dataset = dataset[start:end]
        else:
            # Path should be a local directory.
            dataset = load_dataset(
                path,
                split=split_str,
                # Don't require the number of examples (lines) per split to be pre-defined.
                verification_mode=VerificationMode.NO_CHECKS,
                # But also don't use cached data, as the dataset may have changed on disk.
                download_mode=DownloadMode.FORCE_REDOWNLOAD,
            )

        prompts = list(dataset[specification.column])

    if specification.prefix:
        prompts = [f"{specification.prefix} {prompt}" for prompt in prompts]

    if specification.suffix:
        prompts = [f"{prompt} {specification.suffix}" for prompt in prompts]

    system_prompt = (
        settings.system_prompt
        if specification.system_prompt is None
        else specification.system_prompt
    )

    return [
        Prompt(
            system=system_prompt,
            user=prompt,
        )
        for prompt in prompts
    ]


def batchify(items: list[T], batch_size: int) -> list[list[T]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def get_trial_parameters(trial: Trial | FrozenTrial) -> dict[str, str]:
    params = {}

    direction_index = trial.user_attrs["direction_index"]
    params["direction_index"] = (
        "per layer" if (direction_index is None) else f"{direction_index:.2f}"
    )

    for component, parameters in trial.user_attrs["parameters"].items():
        for name, value in parameters.items():
            params[f"{component}.{name}"] = f"{value:.2f}"

    return params
