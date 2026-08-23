"""Vendored ablation core from p-e-w/heretic (AGPL-3.0-or-later).

Slimmed to the pieces a headless in-process abliteration stage needs:
`config` (pydantic Settings), `model` (loading + LoRA-delta ablation),
`evaluator` (Optuna objective scoring), `scorer` + `scorers` (KeywordRate /
KLDivergence), `plugin` (plugin loading), `utils` (prompt loading / trial
params), `system.empty_cache`, and the headless `abliterate.run_abliteration`.

Dropped upstream: the interactive TUI (questionary), the lm-eval benchmark
suite, Hugging Face upload, reproducibility/archival machinery, the residual
geometry/plotting research module (`analyzer`), multimodal dispatch
(AutoModelForImageTextToText), and the reproducibility helpers.

The headless entry point is `ms_moe_maker.heretic.abliterate.run_abliteration`.
"""
