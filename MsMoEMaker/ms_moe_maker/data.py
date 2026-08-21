"""Data collection pipeline — corpus scraping and MCP trace generation.

Two jobs:
  1. Collect per-language code corpora from HuggingFace shards (adaptive loading).
  2. Generate / collect agentcore MCP traces via teacher-model rejection sampling.

Each function writes a JSONL file and returns its path.  All functions are
idempotent: they skip if the target already exists (unless force=True).

The adaptive shard loader fetches one HuggingFaceCode/stack-v3-train shard at
a time, deduplicates by content hash, and retires each language as soon as it
has enough TOKENS (not documents).  Dedicated language sources (PowerShell,
etc.) bypass the shard scan entirely.
"""
from __future__ import annotations

import json
import os
import random
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# MODULE LEVEL ON PURPOSE. `cfg` used to be imported inside collect_corpus and
# three other functions, and then referenced from _collect_from_shards, which
# imported nothing - so `cfg.safe_name(...)` there was a NameError that only
# fired once a real build reached the shard scan. Killed Chad's first Spark run
# at stage 1.
#
# Same shape as the `import moe_stitch` bug in stitch.py: a module bound LOCAL
# to one function and read as a GLOBAL in another. Python does not complain
# until the line executes, so a rarely-taken branch hides it indefinitely.
#
# config imports only the standard library at module scope, so hoisting it
# costs the no-torch-on-a-laptop promise nothing and removes the whole class of
# failure for this name. Heavy imports (torch, transformers, datasets) stay
# inside their functions, where they belong.
from . import config as cfg
from . import stages as st

# ---------------------------------------------------------------------------
# Corpus collection
# ---------------------------------------------------------------------------

# THE STACK CORPUS, NAMED ONCE. Preflight has to check the same string the
# scan will actually request - a reachability check against a repo id that is
# merely similar to the real one is worse than no check, because it passes.
STACK_REPO = "HuggingFaceCode/stack-v3-train"


def collect_corpus(config, languages: Optional[List[str]] = None,
                   sources: Optional[Dict[str, Any]] = None,
                   callback=None) -> Dict[str, str]:
    """Pull all expert corpora into DATA_ROOT/{safe_name}_code.jsonl.

    Accepts Source info from the recipe so each expert can use its own
    HuggingFace dataset, local dir, or shard-scan target.

    Returns ``{safe_name: path, ...}`` for every language in the recipe.

    ``callback(stage_id, status, note)`` reports progress against a REAL STAGE
    ID from `stages`, with the expert named in the note.

    It used to pass the EXPERT NAME as the stage id - callback("python", ...) -
    and the manifest writer creates a stage for any id it does not recognise,
    so a two-expert run grew phantom "python" and "csharp" stages alongside
    the eight real ones. The count then read 8/9 with nine entries, and any
    viewer painting from the manifest would draw stages that do not exist in
    stages.plan(). Corpus collection is progress WITHIN data.corpus, not a
    stage of its own.
    so the runner can update manifest stage notes.
    """

    sources = sources or {}
    safe_map = {lang: cfg.safe_name(lang) for lang in (languages or [])}
    results = {}

    # ── 1. Handle recipe sources (kind=hf, kind=local) ────────────────────
    for expert_name, src in sources.items():
        safe = safe_map.get(expert_name, cfg.safe_name(expert_name))
        out_path = f"{config.data_root}/{safe}_code.jsonl"

        # Skip if already present (unless forced)
        if not config.force and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            kind = getattr(src, 'kind', '') if hasattr(src, 'kind') else src.get('kind', '')
            print(f"[skip] {expert_name} dataset already present at {out_path}")
            if callback:
                callback(st.DATA_CORPUS, "running", f"{expert_name}: already present")
            results[safe] = out_path
            continue

        kind = getattr(src, 'kind', '') if hasattr(src, 'kind') else src.get('kind', '')

        if kind == "hf":
            repo = getattr(src, 'repo', None) or src.get('repo', '')
            text_field = getattr(src, 'text_field', 'code')
            split = getattr(src, 'split', 'train')
            if not repo:
                print(f"   [skip] {expert_name}: no repo specified")
                continue
            out_path = _collect_hf(repo, text_field, split,
                                   out_path, config, callback, expert_name)
            if out_path:
                results[safe] = out_path

        elif kind == "gh":
            repo = getattr(src, 'repo', None) or src.get('repo', '')
            glob_pat = getattr(src, 'glob', None) or "**/*.md"
            ref = getattr(src, 'ref', None)
            subdir = getattr(src, 'subdir', None)
            if not repo:
                print(f"   [skip] {expert_name}: no repo specified")
                continue
            out_path = _collect_gh(repo, glob_pat, ref, subdir, out_path,
                                   config, callback, expert_name)
            if out_path:
                results[safe] = out_path

        elif kind == "local":
            path = getattr(src, 'path', None) or src.get('path', '')
            glob_pat = getattr(src, 'glob', '**/*.txt') or src.get('glob', '**/*.txt')
            if not path:
                print(f"   [skip] {expert_name}: no path specified")
                continue
            out_path = _collect_local(path, glob_pat, out_path,
                                      config, callback, expert_name)
            if out_path:
                results[safe] = out_path

        elif kind == "stack":
            # Stack sources go through the shard scan (handled below)
            pass

        elif kind == "synth":
            # Synth sources are handled by generate_agent_traces, not corpus
            pass

        else:
            # Unknown kind — fall through to shard scan
            pass

    # ── 2. Handle kind=stack sources ──────────────────────────────────────
    stack_languages = []
    for expert_name, src in sources.items():
        kind = getattr(src, 'kind', '') if hasattr(src, 'kind') else src.get('kind', '')
        if kind == "stack":
            lang = getattr(src, 'language', None) or src.get('language', expert_name)
            stack_languages.append(lang)

    # ── 3. Fall-through languages (not covered by sources) ────────────────
    remaining_languages = list(safe_map.keys())  # expert names from recipe
    if stack_languages:
        # Stack: scan shards filtered by the specified language(s)
        print(f"\nShard scan for stack sources: {stack_languages}")
        scanned = _collect_from_shards(stack_languages, config, callback)
        results.update(scanned)

    if remaining_languages:
        # Normalize to CODE_LANGUAGES for shard scan (case fix: "python" → "Python")
        normalized = []
        name_lookup = {}  # CODE_LANG → expert_name for safe_map
        for expert_name in remaining_languages:
            safe = safe_map.get(expert_name, cfg.safe_name(expert_name))
            # Map to CODE_LANGUAGES if possible
            mapped = _expert_to_code_lang(expert_name)
            if mapped and mapped not in [n for n in normalized]:
                normalized.append(mapped)
                name_lookup[mapped] = expert_name
            elif not normalized:  # not in CODE_LANGUAGES, still scan
                normalized.append(cfg.safe_name(expert_name))
                name_lookup[cfg.safe_name(expert_name)] = expert_name

        if normalized:
            scanned = _collect_from_shards(normalized, config, callback)
            # Re-map keys from CODE_LANG to expert name
            for code_lang, path in scanned.items():
                safe = safe_map.get(name_lookup.get(code_lang, code_lang),
                                    cfg.safe_name(code_lang))
                results[safe] = path

    return results


def _expert_to_code_lang(expert_name: str) -> Optional[str]:
    """Map an expert name to a CODE_LANGUAGE if possible.

    'python' → 'Python', 'csharp' → 'C#', etc.
    Returns None if no mapping exists.
    """
    # Reverse lookup: lowercase expert name → CODE_LANGUAGE
    for lang in cfg.CODE_LANGUAGES:
        if cfg.safe_name(lang) == expert_name.lower():
            return lang
    return None


def _collect_from_dedicated(lang: str, source: Any,
                            config, callback=None) -> Optional[str]:
    """Pull one language from its own corpus (HuggingFace dataset or local)."""
    import sys

    safe = cfg.safe_name(lang)
    out_path = f"{config.data_root}/{safe}_code.jsonl"
    if not config.force and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"[skip] {lang} dataset already present at {out_path}")
        return out_path

    print(f"\n[{lang}] collecting from dedicated source")

    if hasattr(source, 'kind') and source.kind == "local":
        # Local files
        return _collect_local(source.path, source.glob, out_path, config, callback, lang)

    if hasattr(source, 'kind') and source.kind == "hf":
        return _collect_hf(source.repo, source.text_field, source.split,
                           out_path, config, callback, lang)

    # Default: try to load as HF dataset
    repo = source.get("repo", source.get("repo_id", "")) if isinstance(source, dict) else str(source)
    if repo:
        return _collect_hf(repo, source.get("text_field", "code"),
                           source.get("split", "train"),
                           out_path, config, callback, lang)

    return None


def _collect_hf(repo: str, text_field: str, split: str,
                out_path: str, config, callback=None, lang: str = "") -> Optional[str]:
    """Load a HuggingFace dataset and write JSONL."""
    try:
        from datasets import load_dataset
    except ImportError:
        print(f"   [skip] {lang}: datasets package not installed")
        return None

    ds = load_dataset(repo, split=split, cache_dir=config.hf_home)

    # Schema assertion
    cols = set(ds.column_names)
    if text_field not in cols:
        print(f"   ERROR: {repo} has no {text_field!r} column — available: {sorted(cols)}")
        return None
    print(f"   {len(ds)} rows, columns {sorted(cols)}")

    kept, seen, kept_chars = [], set(), 0
    stop_reason = "source exhausted"

    for row in ds:
        content = row.get(text_field) or ""
        if len(content) < 10:
            continue
        nlines = content.count("\n")
        if nlines < 3 or nlines > 10000:
            continue
        h = hashlib.sha1(content.encode("utf-8", "ignore")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        kept.append({"text": content, "repo": repo})
        kept_chars += len(content)
        if kept_chars / config.chars_per_token_est >= config.collect_token_target:
            stop_reason = "token target met"
            break
        if len(kept) >= config.num_code_samples:
            stop_reason = f"hit the {config.num_code_samples}-document ceiling"
            break

    if len(kept) < config.min_samples_per_expert:
        print(f"   ERROR: {lang}: only {len(kept)} samples from {repo} "
              f"(min {config.min_samples_per_expert})")
        return None

    with open(out_path, "w", encoding="utf-8") as f:
        for item in kept:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"   [{lang}] {len(kept)} samples, ~{kept_chars/config.chars_per_token_est/1e6:.1f}M "
          f"est. tokens → {out_path}  ({stop_reason})")
    if callback:
        callback(st.DATA_CORPUS, "running", f"{lang}: {len(kept)} samples")
    return out_path


def _glob_to_regex(pattern: str) -> "re.Pattern":
    """Translate a path glob to a regex with REAL `**` semantics.

    fnmatch is the obvious tool here and it is the wrong one: it has no
    concept of a path separator, so `*` happily eats `/` and `**` means
    nothing in particular. The visible consequence is that `docs/**/*.md` -
    which every developer reads as "markdown anywhere under docs, including
    directly in it" - does NOT match `docs/README.md` under fnmatch, because
    the pattern demands a literal slash between the two wildcards.

    That failure is silent and it looks like an empty repository. Caught by
    the tests here, which is what tests on a fetcher are for.

    The rules, matching what pathlib and every shell mean by them:
        **/  zero or more directories   -> (?:.*/)?
        **   anything, slashes included -> .*
        *    anything within ONE segment-> [^/]*
        ?    one character, not a slash -> [^/]
    """
    import re as _re
    out = []
    i, n = 0, len(pattern)
    while i < n:
        c = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")      # zero or more directories
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(_re.escape(c))
            i += 1
    return _re.compile("^" + "".join(out) + "$")


def _fetch_gh_tarball(repo: str, ref: Optional[str]) -> Tuple[Optional[bytes], str]:
    """Download a whole ref as one .tar.gz. Returns (bytes, ref-used).

    ONE TARBALL, NOT A CLONE. codeload serves a ref as a single archive, so
    this is one request with no git binary on the box and no history
    downloaded - for a docs repo the history is usually most of the bytes and
    none of the corpus.

    The default ref is HEAD on purpose: codeload resolves the repo's own
    default branch, so nobody has to know whether a project says `main` or
    `master`. Guessing `main` and 404ing on an older repo is the kind of
    papercut that reads as "your tool is broken".

    Public repos only, deliberately. A token would have to be read from the
    environment, stored in a recipe, or prompted for - and a recipe is a
    document people SHARE, which makes it the wrong object to put a credential
    in.

    Split out from the parsing half so the parsing half can be tested without
    a network. That is not hypothetical tidiness: this function could not be
    exercised where it was written, so everything it does NOT do now lives in
    _corpus_from_tarball, which can be.
    """
    import urllib.error
    import urllib.request

    url = f"https://codeload.github.com/{repo}/tar.gz/{ref or 'HEAD'}"
    req = urllib.request.Request(url, headers={
        # GitHub is entitled to refuse an anonymous client with no identity.
        "User-Agent": "ms-moe-maker (+https://github.com/ChadRoesler/MsMoEMaker)",
        "Accept": "application/x-gzip",
    })
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read(), (ref or "HEAD")
    except urllib.error.HTTPError as exc:
        hint = ""
        if exc.code == 404:
            hint = (" - check owner/name and the ref; private repos are not "
                    "supported")
        elif exc.code in (403, 429):
            hint = (" - rate-limited or blocked. codeload allows anonymous "
                    "downloads, so this is usually a network policy between "
                    "you and GitHub rather than the repo")
        print(f"   gh {repo}@{ref or 'HEAD'}: HTTP {exc.code}{hint}")
    except (urllib.error.URLError, OSError) as exc:
        print(f"   gh {repo}@{ref or 'HEAD'}: {exc}")
    return None, ""


def _corpus_from_tarball(tar_bytes: bytes, glob_pattern: str,
                         subdir: Optional[str], config) -> List[Dict[str, str]]:
    """Turn a codeload tarball into corpus rows. No network, so: testable."""
    import io
    import tarfile

    matches = _glob_to_regex(glob_pattern)
    kept: List[Dict[str, str]] = []
    seen = set()
    kept_chars = 0

    try:
        tf = tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz")
    except tarfile.TarError as exc:
        print(f"   ERROR: not a readable tarball: {exc}")
        return []

    with tf:
        for member in tf:
            if not member.isfile():
                continue
            # codeload prefixes every path with "<name>-<sha>/". Strip it, so a
            # recipe's glob is written against the REPO and not against a
            # directory name that changes with every commit.
            parts = member.name.split("/", 1)
            rel = parts[1] if len(parts) == 2 else parts[0]
            if subdir:
                pre = subdir.strip("/") + "/"
                if not rel.startswith(pre):
                    continue
                rel = rel[len(pre):]
            if not matches.match(rel):
                continue
            if member.size > 2_000_000:      # a 2 MB "text" file is a blob
                continue
            fh = tf.extractfile(member)
            if fh is None:
                continue
            try:
                content = fh.read().decode("utf-8", "ignore")
            except OSError:
                continue

            if len(content) < 10:
                continue
            nlines = content.count("\n")
            if nlines < 3 or nlines > 10000:
                continue
            h = hashlib.sha1(content.encode("utf-8", "ignore")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            # The tarball's top-level directory is the repo. `repo` itself
            # is not in scope here on purpose - this function takes bytes so
            # it can be tested without a network.
            kept.append({"text": content,
                         "repo": member.name.split("/", 1)[0],
                         "path": member.name, "lang": "gh"})
            kept_chars += len(content)
            if (config.collect_token_target
                    and kept_chars / config.chars_per_token_est
                    >= config.collect_token_target):
                break
            if len(kept) >= config.num_code_samples:
                break
    return kept


def _collect_gh(repo: str, glob_pattern: str, ref: Optional[str],
                subdir: Optional[str], out_path: str,
                config, callback=None, lang: str = "") -> Optional[str]:
    """Pull text files out of a public GitHub repo into a corpus JSONL."""
    if "/" not in repo:
        print(f"   ERROR: gh repo {repo!r} must be owner/name")
        return None

    tar_bytes, used_ref = _fetch_gh_tarball(repo, ref)
    if tar_bytes is None:
        print(f"   ERROR: could not fetch {repo} (public repos only)")
        return None
    print(f"   [{lang}] fetched {repo}@{used_ref} "
          f"({len(tar_bytes) / 2**20:.1f} MB)")

    kept = _corpus_from_tarball(tar_bytes, glob_pattern, subdir, config)

    if len(kept) < config.min_samples_per_expert:
        print(f"   ERROR: gh: only {len(kept)} samples from {repo} matching "
              f"{glob_pattern!r} (min {config.min_samples_per_expert}). The "
              f"glob is matched against paths RELATIVE to the repo root, so "
              f"'docs/**/*.md', not '**/docs/**/*.md'.")
        return None

    with open(out_path, "w", encoding="utf-8") as f:
        for item in kept:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"   [{lang}] {len(kept)} samples from {repo} → {out_path}")
    if callback:
        callback(st.DATA_CORPUS, "running", f"{lang}: {len(kept)} samples from {repo}")
    return out_path


def _collect_local(path: str, glob_pattern: str, out_path: str,
                   config, callback=None, lang: str = "") -> Optional[str]:
    """Read local text files matching glob and write JSONL."""
    import glob as globmod

    if not os.path.exists(path):
        print(f"   ERROR: local path {path} does not exist")
        return None

    kept, seen, kept_chars = [], set(), 0

    for filepath in globmod.glob(os.path.join(path, glob_pattern), recursive=True):
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except (OSError, IOError):
            continue

        if len(content) < 10:
            continue
        nlines = content.count("\n")
        if nlines < 3 or nlines > 10000:
            continue
        h = hashlib.sha1(content.encode("utf-8", "ignore")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        kept.append({"text": content, "repo": path,
                     "path": str(filepath), "lang": "local"})
        kept_chars += len(content)
        if kept_chars / config.chars_per_token_est >= config.collect_token_target:
            break
        if len(kept) >= config.num_code_samples:
            break

    if len(kept) < config.min_samples_per_expert:
        print(f"   ERROR: local: only {len(kept)} samples from {path} "
              f"(min {config.min_samples_per_expert})")
        return None

    with open(out_path, "w", encoding="utf-8") as f:
        for item in kept:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"   [{lang}] {len(kept)} local samples → {out_path}")
    if callback:
        callback(st.DATA_CORPUS, "running", f"{lang}: {len(kept)} local samples")
    return out_path


def _repo_label(repo: Dict[str, Any], fallback: str = "") -> str:
    """A name for the repo this shard row represents.

    The row IS the repo, so IDENTITY never depends on finding a name - the
    per-repo cap counts within a row and is correct regardless. This is only
    the LABEL, and a label that invents a repo name is worse than one that
    admits it does not have one.

    The first version took the first two components of files[0].file_path,
    which is owner/name in a github-shaped corpus and nonsense here: a
    markdown file at repo root has no directory at all, so every project's
    README collapsed into a single fake repo called "README.md" and the
    report announced that 26% of the corpus came from it. The stack schema is
    (content, content_id, file_path, language) - there IS no repo name in it.

    So: a real name field if one exists; else the longest common directory
    prefix across this repo's files, which is a genuine signature of a repo
    laid out in subdirectories; else the synthetic row id, which is honest
    about being an id rather than a name.
    """
    # repo_path FIRST, because the corpus actually has it and two versions of
    # this function guessed instead. The row-keys line printed by the schema
    # check exists for exactly this: the shard rows carry
    #   ['commit_id', 'files', 'github_metadata', 'num_files', 'repo_id',
    #    'repo_path']
    # and every key this used to look for was absent, so it fell through to a
    # path heuristic and reported the result as a repo name. Guessing at a
    # field is a bug; guessing when the real field is one print statement away
    # is an avoidable one.
    for key in ("repo_path", "repo_name", "repository", "repo", "repo_id",
                "name", "id", "max_stars_repo_name"):
        v = repo.get(key)
        if isinstance(v, str) and v:
            return v

    files = repo.get("files") or []
    dirs = []
    for f in files[:64]:
        path = (f.get("file_path") or "").strip("/")
        parts = [p for p in path.split("/") if p][:-1]      # drop the filename
        if parts:
            dirs.append(parts)
    if dirs:
        common = dirs[0]
        for parts in dirs[1:]:
            keep = 0
            for a, b in zip(common, parts):
                if a != b:
                    break
                keep += 1
            common = common[:keep]
            if not common:
                break
        if common:
            return "/".join(common[:2])
    return fallback or "<row>"


def _diversity(counter, total: int) -> Tuple[int, str, float]:
    """(distinct repos, biggest contributor, its share of the corpus)."""
    if not counter or not total:
        return (0, "", 0.0)
    name, n = counter.most_common(1)[0]
    return (len(counter), name, n / total)


def _line_reuse(docs: List[Dict[str, Any]], sample: int = 300) -> float:
    """Fraction of non-blank lines that are repeats, over a sample.

    The cheap tell for a templated corpus. On a real build this separated a
    C# bucket at 82.4% from a Python bucket at 34.1% - and the 82.4% one was
    a single company's application, where 78% of files opened with the same
    proprietary `using` and every method was wrapped in the same trace-log
    call. It trained beautifully and learned a house style.
    """
    lines: List[str] = []
    for d in docs[:sample]:
        for line in (d.get("text") or "").splitlines():
            line = line.strip()
            if line:
                lines.append(line)
    if not lines:
        return 0.0
    return 1.0 - (len(set(lines)) / len(lines))


def _collect_from_shards(languages: List[str], config,
                         callback=None) -> Dict[str, str]:
    """Adaptive shard scan — fetch stack-v3-train shards until every language is satisfied."""
    from huggingface_hub import hf_hub_download, list_repo_files
    from collections import Counter

    cap = min(config.max_shards, 1000)  # safe cap
    safe_map = {l: cfg.safe_name(l) for l in languages}

    # Check if everything is already on disk
    existing = {l: f"{config.data_root}/{safe_map[l]}_code.jsonl" for l in languages}
    if not config.force and all(os.path.exists(p) and os.path.getsize(p) > 0 for p in existing.values()):
        print("[skip] code datasets already built:")
        for l, p in existing.items():
            with open(p) as fh:
                n = sum(1 for _ in fh)
            print(f"          {l:12} {n:>7} samples  ({p})")
        return {safe_map[l]: p for l, p in existing.items()}

    repo_id = STACK_REPO
    all_files = list_repo_files(repo_id, repo_type="dataset")
    shard_files = sorted(f for f in all_files if f.startswith("data/part-") and f.endswith(".parquet"))
    if not shard_files:
        raise RuntimeError(f"no data/part-*.parquet shards found in {repo_id}")

    cap = min(config.max_shards, len(shard_files))
    print(f"{len(shard_files)} shards available; will pull at most {cap} "
          f"(~{cap * 0.57:.0f} GB) until every language holds "
          f"~{config.collect_token_target/1e6:.1f}M est. tokens")

    # ONE REPO MUST NOT BE THE CORPUS.
    #
    # The quota is in TOKENS and a single large repository can satisfy it
    # before the scan ever reaches a second one. Measured: a C# bucket hit its
    # 1.8M-token target from 658 files in shard 1, of which 515 imported the
    # same proprietary namespace - one Japanese enterprise application, 78% of
    # the corpus, held-out perplexity 1.33. Every downstream check passed. The
    # expert trained, diverged from its neighbour at 263x chance, and won its
    # own domain by 0.43 nats. It was an expert in one company's house style
    # and nothing in the pipeline could tell.
    #
    # Verbose languages are where this bites: C#, Java, Go with generated
    # bindings, anything with a codegen culture. Python needed 1754 files for
    # the same token count and got real spread for free.
    #
    # The cap is per repo PER LANGUAGE, so a monorepo that legitimately holds
    # both still contributes to both - it just cannot BE either.
    per_repo_cap = getattr(config, "per_repo_cap", 0) or 20

    buckets = {lang: [] for lang in languages}
    chars = {lang: 0 for lang in languages}
    seen = {lang: set() for lang in languages}
    repos_for = {lang: Counter() for lang in languages}
    capped_hits = {lang: 0 for lang in languages}
    done = set()
    repos_scanned = 0
    shards_used = 0
    schema_checked = False

    for shard_no, fname in enumerate(shard_files[:cap], 1):
        hunting = [l for l in languages if l not in done]
        print(f"\n--- shard {shard_no}/{cap}  (still hunting: {hunting})")
        local = hf_hub_download(repo_id, fname, repo_type="dataset",
                                cache_dir=config.shard_cache)

        try:
            from datasets import load_dataset
            ds = load_dataset("parquet", data_files=[local], split="train")
        except Exception as exc:
            print(f"   failed to load shard {shard_no}: {exc}")
            continue

        shards_used = shard_no

        if not schema_checked:
            probe = ds[0]
            if "files" not in probe:
                raise RuntimeError(f"expected a 'files' column; got {sorted(probe.keys())}")
            fkeys = set(probe["files"][0].keys())
            required = {"content", "file_path", "language", "content_id"}
            missing = required - fkeys
            if missing:
                raise RuntimeError(f"schema drift: files[] missing {sorted(missing)}; "
                                   f"actual fields {sorted(fkeys)}")
            print(f"    [schema] files[] OK: {sorted(required)}")
            # ROW-LEVEL KEYS, PRINTED. The repo label was guessed from
            # file_path for want of knowing what else was on the row, and the
            # guess was wrong in a way that printed as a finding. One line of
            # output ends that permanently.
            print(f"    [schema] row keys: {sorted(probe.keys())}")

            # Language census
            census = Counter()
            for row in ds.select(range(min(200, len(ds)))):
                for f in row["files"]:
                    census[f["language"]] += 1
            print(f"    [langs] top names: {census.most_common(12)}")
            absent = [l for l in languages if l not in census]
            if absent:
                print(f"    [langs] NOTE {absent} unseen in 200 repos — rare or misspelled")
            schema_checked = True

        for repo in ds:
            repos_scanned += 1
            rlabel = _repo_label(repo, fallback=f"shard{shard_no}#{repos_scanned}")
            taken_here = {lang: 0 for lang in languages}
            for f in repo.get("files", []):
                lang = f.get("language")
                if lang not in languages or lang in done:
                    continue
                if f.get("is_vendor"):
                    continue
                if taken_here[lang] >= per_repo_cap:
                    capped_hits[lang] += 1
                    continue
                cid = f.get("content_id")
                if not cid or cid in seen[lang]:
                    continue
                content = f.get("content") or ""
                if len(content) < 10:
                    continue
                nlines = content.count("\n")
                if nlines < 3 or nlines > 10000:
                    continue
                seen[lang].add(cid)
                # PROVENANCE IS PART OF THE CORPUS, NOT A DEBUG AID.
                #
                # These rows used to be {"text": ...} and nothing else, so a
                # finished corpus could not answer the one question that
                # mattered about it: how much of this came from one project?
                # The collector knew - it was iterating repos - and threw the
                # answer away at write time. That is why the single-repo C#
                # corpus could only be diagnosed by inference from line reuse,
                # and why it could not be repaired after the fact at all.
                #
                # Purely additive: everything downstream reads `text`.
                buckets[lang].append({"text": content, "repo": rlabel,
                                      "path": f.get("file_path") or "",
                                      "lang": lang})
                chars[lang] += len(content)
                taken_here[lang] += 1
                repos_for[lang][rlabel] += 1

                # TWO UNITS, AND BOTH HAVE TO BE SATISFIED BEFORE WE STOP.
                #
                # The scan retires a language on TOKENS - that is the unit the
                # training schedule is actually denominated in. min_samples is
                # a floor on DOCUMENTS, and it used to be checked only AFTER
                # the loop had already broken out on "All languages
                # satisfied". So a run with min_samples above the doc count
                # the token target happens to need would stop hunting the
                # moment tokens were met, then fail a check it had stopped
                # trying to satisfy - and raising max_shards did nothing,
                # because the loop never reached another shard.
                #
                # Measured: `[Python] FULL at ~7.4M est. tokens (6778 docs)`
                # immediately followed by `buckets below min 9000`. The scan
                # was not stuck; it had declared success by one measure and
                # been failed by another.
                #
                # A floor that cannot steer the loop is not a floor, it is a
                # late assertion. Both now gate `done`.
                est_tok = chars[lang] / config.chars_per_token_est
                have_tokens = est_tok >= config.collect_token_target
                have_docs = len(buckets[lang]) >= config.min_samples_per_expert
                if have_tokens and have_docs:
                    done.add(lang)
                    print(f"    [{lang}] FULL at ~{est_tok/1e6:.1f}M est. tokens "
                          f"({len(buckets[lang])} docs, shard {shard_no})")
                elif len(buckets[lang]) >= config.num_code_samples:
                    # The ceiling wins over both - it exists so a pathological
                    # language cannot eat the disk, and stopping is the point.
                    done.add(lang)
                    print(f"    [{lang}] hit the {config.num_code_samples}-doc "
                          f"CEILING with ~{est_tok/1e6:.1f}M est. tokens"
                          + ("" if have_docs else
                             f" - still under the {config.min_samples_per_expert}-doc "
                             f"floor, which the ceiling cannot be raised past"))
                elif have_tokens and not have_docs and shard_no == 1:
                    # Say it the FIRST time it happens, not after 80 shards.
                    print(f"    [{lang}] token target met at "
                          f"{len(buckets[lang])} docs; still hunting for the "
                          f"{config.min_samples_per_expert}-doc floor")

        del ds
        import gc
        gc.collect()
        print(f"    after shard {shard_no}: { {l: len(buckets[l]) for l in languages} }")
        if len(done) == len(languages):
            print("\nAll languages satisfied — stopping early.")
            break

    print(f"\nScanned {repos_scanned} repos across {shards_used} shard(s).")
    health: Dict[str, Dict[str, Any]] = {}
    for lang in languages:
        n = len(buckets[lang])
        est = chars[lang] / config.chars_per_token_est
        nrepos, top_repo, top_share = _diversity(repos_for[lang], n)
        reuse = _line_reuse(buckets[lang])
        health[lang] = {"docs": n, "est_tokens": est, "repos": nrepos,
                        "top_repo": top_repo, "top_repo_share": top_share,
                        "line_reuse": reuse, "capped": capped_hits[lang]}
        print(f"   {lang}: {n} docs, ~{est/1e6:.1f}M est. tokens, "
              f"{nrepos} repos "
              f"(largest {top_share:.1%}), line reuse {reuse:.0%}"
              + (f", {capped_hits[lang]} files skipped by the "
                 f"{per_repo_cap}/repo cap" if capped_hits[lang] else ""))

        # WARN, DO NOT REFUSE. Someone may want one codebase on purpose -
        # that is a legitimate expert. They should just never get it by
        # accident, which is what happened.
        if nrepos and top_share > 0.25:
            print(f"   [warn] {lang}: {top_share:.0%} of this corpus is one "
                  f"repo ({top_repo}). The expert will learn that project's "
                  f"house style as if it were the language. Raise "
                  f"corpus.per_repo_cap's strictness or widen the source.")
        if reuse > 0.7:
            print(f"   [warn] {lang}: {reuse:.0%} of lines are repeats - this "
                  f"reads as generated or templated code. Expect a very low "
                  f"held-out loss that means memorised form, not fluency.")

    # Minimum viable bucket check
    starved = {l: len(buckets[l]) for l in languages if len(buckets[l]) < config.min_samples_per_expert}
    if starved:
        # SAY WHICH LIMIT ACTUALLY STOPPED IT. "Raise max_shards" was the only
        # advice offered and it was the wrong advice in the common case: the
        # scan usually stops because it ran out of SHARDS, but it can also
        # stop because the corpus genuinely does not hold that much of this
        # language. Those need opposite responses and the message could not
        # tell them apart.
        ran_out_of_shards = shards_used >= cap
        detail = []
        for l, n in starved.items():
            tok = chars[l] / config.chars_per_token_est
            detail.append(f"{l}: {n} docs (~{tok/1e6:.1f}M est. tokens)")
        raise RuntimeError(
            f"corpus.min_samples is {config.min_samples_per_expert} docs and "
            f"these did not reach it after {shards_used} shard(s) — "
            + "; ".join(detail) + ". "
            + (f"The {cap}-shard cap (corpus.max_shards) is what stopped the "
               f"scan: raise it and the scan will keep hunting. "
               if ran_out_of_shards else
               f"The scan stopped before the {cap}-shard cap, so more shards "
               f"were available and something else retired these languages — "
               f"check the doc CEILING (corpus.max_samples="
               f"{config.num_code_samples}). ")
            + f"Note min_samples is a DOC floor while the training budget is "
              f"in TOKENS: at target_steps this run needs "
              f"~{config.collect_token_target/1e6:.1f}M tokens per expert, "
              f"which these languages reach at fewer documents than "
              f"{config.min_samples_per_expert}. If the token budget is what "
              f"you care about, lower min_samples; it is a floor against "
              f"training on scraps, not a way to ask for more data.")

    # Write JSONL
    paths = {}
    for lang in languages:
        safe = safe_map[lang]
        out_path = f"{config.data_root}/{safe}_code.jsonl"
        with open(out_path, "w", encoding="utf-8") as fh:
            for s_ in buckets[lang][:config.num_code_samples]:
                fh.write(json.dumps(s_, ensure_ascii=False) + "\n")
        print(f"Saved {safe} → {out_path} ({len(buckets[lang][:config.num_code_samples])} samples)")
        # Measure the file we just wrote, not the buckets we just held. Same
        # module `validate` and `ms-moe-maker corpus` use, so a reader never
        # has to wonder whether the build's numbers and the checker's numbers
        # came from the same arithmetic.
        try:
            from . import corpushealth as _ch
            print(_ch.format_health(_ch.inspect(out_path)))
        except Exception as exc:            # advisory: never fail the stage
            print(f"   [warn] corpus health check skipped: {exc}")
        if callback:
            callback(st.DATA_CORPUS, "running", f"{lang}: {len(buckets[lang])} docs")
        paths[safe] = out_path
        buckets[lang].clear()  # free memory

    return paths


# ---------------------------------------------------------------------------
# MCP Agent trace generation
# ---------------------------------------------------------------------------

def generate_agent_traces(config, callback=None) -> Optional[str]:
    """Generate MCP agent traces for the agentcore expert.

    Uses rejection sampling: a teacher model generates tool calls, which are
    validated against a freshly-generated tool surface.  Server-agnostic by
    default (fresh synthetic surface each example).

    Returns path to the final JSONL, or None if generation was skipped.
    """
    import sys

    out_path = f"{config.data_root}/agentcore_code.jsonl"
    if not config.force and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"[skip] agent dataset already present at {out_path}")
        return out_path

    # Check if the base model has a chat template
    base_model = config.base if config.base else config.base_safe
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=config.hf_home)
        if not getattr(tokenizer, "chat_template", None):
            raise RuntimeError(
                f"{base_model} has no chat_template. The agent dataset requires "
                f"an Instruct-family base model. Use an Instruct model or set "
                f"a custom tokenizer.chat_template.")
    except Exception as exc:
        print(f"WARNING: cannot check chat_template: {exc}")
        # Continue anyway — the template check happens later.

    mcp_path = f"{config.data_root}/agentcore_mcp_code.jsonl"
    print(f"\nGenerating MCP agent traces "
          f"{'via vLLM' if config.use_vllm else 'via transformers+bitsandbytes'}")

    # Resume from partial
    partial_path = mcp_path + ".partial"
    kept = 0
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            kept = sum(1 for line in f if line.strip())
        if kept >= config.num_agent_samples:
            os.replace(partial_path, out_path)
            print(f"   resumed a complete .partial ({kept}) → {out_path}")
            if callback:
                callback(st.DATA_SYNTH, "running", f"agentcore: {kept} traces")
            return out_path
        print(f"   resuming from {partial_path} with {kept} traces already banked")

    # Load tool surface
    real_surface = _load_tool_surface()

    system_prompt = (
        "You are a universal agent that communicates with any MCP server via JSON-RPC 2.0.\n"
        'Discover tools with: {"jsonrpc":"2.0","id":1,"method":"tools/list"}\n'
        'Call one with: {"jsonrpc":"2.0","id":2,"method":"tools/call",'
        '"params":{"name":"<tool>","arguments":{...}}}\n'
        "Use ONLY tools present in the tools/list result, and obey their inputSchema exactly."
    )

    # Build teacher
    if config.use_vllm:
        teacher = _VLLMTeacher(config)
    else:
        teacher = _HFTeacher(config)

    tokenizer = teacher.tokenizer
    attempted, rejects = 0, 0
    t0 = time.time()
    t_kept0 = kept

    sink = open(partial_path, "a", buffering=1)
    try:
        while kept < config.num_agent_samples:
            n = min(teacher.batch_size, (config.num_agent_samples - kept) * 2)
            batch_specs = _agent_batch_specs(n, real_surface, system_prompt)
            prompts = [tokenizer.apply_chat_template(m, tokenize=False,
                                                     add_generation_prompt=True)
                       for m, _, _, _ in batch_specs]
            completions = teacher.complete(prompts)

            for (msgs, tools, task, listing), text in zip(batch_specs, completions):
                attempted += 1
                by_name = {t["name"]: t for t in tools}
                calls = _extract_calls(text)
                calls = [c for c in calls if c.get("method") == "tools/call"]

                if not calls:
                    rejects += 1
                    continue

                verdicts = [_validate_tool_call(c, by_name) for c in calls]
                bad = [why for ok, why in verdicts if not ok]
                if bad:
                    rejects += 1
                    continue

                conv = msgs + [{"role": "assistant", "content": text}]
                text_out = tokenizer.apply_chat_template(conv, tokenize=False) + tokenizer.eos_token
                sink.write(json.dumps({"text": text_out}, ensure_ascii=False) + "\n")
                kept += 1
                if kept >= config.num_agent_samples:
                    break

            # Progress and accept rate
            session = kept - t_kept0
            rate = session / max(time.time() - t0, 1e-9)
            acc = 100.0 * session / max(attempted, 1)
            eta = (config.num_agent_samples - kept) / max(rate, 1e-9) / 60
            print(f"   kept {kept}/{config.num_agent_samples}  "
                  f"(accept {acc:.0f}% of {attempted}, {rate:.1f}/s, ETA {eta:.0f} min)")

            # Tripwire: broken teacher
            if attempted > 200 and acc < 5:
                raise RuntimeError(
                    f"accept rate {acc:.1f}% after {attempted} attempts — the teacher is not "
                    f"producing valid calls. Top rejections: {rejects}")
    finally:
        sink.close()
        teacher.close()

    os.replace(partial_path, out_path)
    print(f"Generated {kept} VALIDATED MCP traces → {out_path}")
    acc_rate = 100.0 * (kept - t_kept0) / max(attempted, 1)
    print(f"   accept rate {acc_rate:.1f}%  top rejections: {rejects}")
    if callback:
        callback(st.DATA_SYNTH, "running", f"agentcore: {kept} traces")
    return out_path


def _load_tool_surface() -> Optional[List[Dict[str, Any]]]:
    """Load optional real tool surface for specialisation."""
    import os
    url = os.environ.get("SEREN_TOOL_SURFACE_URL", "")
    filepath = os.environ.get("SEREN_TOOL_SURFACE_FILE", "")

    if not url and not filepath:
        return None

    tools = None
    if url:
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=5) as r:
                payload = json.loads(r.read().decode())
            tools = payload.get("tools", payload) if isinstance(payload, dict) else payload
        except Exception as e:
            print(f"   tool surface URL failed ({e})")

    if tools is None and filepath and os.path.exists(filepath):
        with open(filepath) as f:
            tools = json.load(f)

    if not tools:
        raise RuntimeError("a tool surface was requested but could not be loaded")

    for t in tools:
        t["inputSchema"] = t.get("inputSchema") or t.get("input_schema") or {}
    usable = [t for t in tools if t.get("name") and t["inputSchema"]]
    print(f"   SPECIALISING on {len(usable)} real tools (agnostic mode is off)")
    return usable


def _synth_tool_surface(rnd, n_tools: int) -> List[Dict[str, Any]]:
    """Freshly generated MCP tool surface — server-agnostic."""
    _VERBS = ["get", "list", "create", "delete", "update", "search", "fetch", "send",
              "start", "stop", "restart", "check", "validate", "export", "import",
              "resolve", "register", "purge", "archive", "summarise", "count", "diff"]
    _NOUNS = ["record", "job", "device", "session", "index", "artifact", "snapshot",
              "queue", "bucket", "ticket", "node", "channel", "policy", "token",
              "manifest", "schedule", "profile", "endpoint", "volume", "alert"]
    _ADJS = ["stale", "pending", "archived", "active", "orphaned", "recent"]
    _STRARG = ["name", "path", "key", "query", "target", "label", "region", "id",
               "pattern", "prefix", "channel", "format", "source", "destination"]
    _INTARG = ["limit", "offset", "depth", "timeout_seconds", "retries", "port",
               "max_results", "page", "priority", "ttl_hours"]
    _BOOL = ["recursive", "dry_run", "force", "verbose", "include_archived",
             "follow_symlinks", "strict"]
    _ENUMS = {
        "format": ["json", "yaml", "csv", "text"],
        "level":  ["debug", "info", "warn", "error"],
        "order":  ["asc", "desc"],
        "scope":  ["local", "global", "session"],
        "state":  ["open", "closed", "any"],
    }
    _PATTERNS = [
        ("path", r"^/[A-Za-z0-9._/-]*$"),
        ("id",   r"^[a-f0-9]{8}$"),
        ("name", r"^[a-z][a-z0-9_]{2,30}$"),
    ]

    tools, used = [], set()
    for _ in range(n_tools):
        for _try in range(20):
            name = f"{rnd.choice(_VERBS)}_{rnd.choice(_NOUNS)}"
            if rnd.random() < 0.25:
                name = f"{rnd.choice(_VERBS)}_{rnd.choice(_ADJS)}_{rnd.choice(_NOUNS)}"
            if name not in used:
                used.add(name)
                break
        else:
            continue

        props, required = {}, []
        for _ in range(rnd.randint(1, 4)):
            roll = rnd.random()
            if roll < 0.40:
                a = rnd.choice(_STRARG)
                spec = {"type": "string", "description": f"The {a} to operate on."}
                if rnd.random() < 0.30:
                    for key, pat in _PATTERNS:
                        if key in a:
                            spec["pattern"] = pat
                            break
                props[a] = spec
            elif roll < 0.62:
                a = rnd.choice(_INTARG)
                spec = {"type": "integer", "description": f"Value for {a}."}
                if rnd.random() < 0.5:
                    spec["minimum"] = 1
                    spec["maximum"] = rnd.choice([10, 100, 1000])
                props[a] = spec
            elif roll < 0.78:
                a = rnd.choice(_BOOL)
                props[a] = {
                    "type": "boolean",
                    "description": f"Whether to {a.replace('_', ' ')}."
                }
            elif roll < 0.92:
                a = rnd.choice(list(_ENUMS))
                props[a] = {
                    "type": "string",
                    "enum": list(_ENUMS[a]),
                    "description": f"One of {', '.join(_ENUMS[a])}."
                }
            else:
                a = rnd.choice(_STRARG) + "s"
                props[a] = {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"A list of {a}."
                }

        if props:
            for a in props:
                if rnd.random() < 0.5:
                    required.append(a)
            if not required:
                required = [rnd.choice(list(props))]

        desc_suffix = rnd.choice([
            "Returns a summary.",
            "Operates on the given target.",
            "Idempotent.",
            "May take a moment.",
        ])
        tools.append({
            "name": name,
            "description": f"{name.replace('_', ' ').capitalize()}. {desc_suffix}",
            "inputSchema": {"type": "object", "properties": props, "required": required},
        })
    return tools


def _agent_batch_specs(n: int, real_surface, system_prompt: str):
    """n independent (msgs, tools, task, listing) specs."""
    batch = []
    for _ in range(n):
        k = random.randint(3, 8)
        if real_surface:
            tools = random.sample(real_surface, min(k, len(real_surface)))
        else:
            tools = _synth_tool_surface(random, k)

        target = random.choice(tools)
        listing = {
            "jsonrpc": "2.0", "id": 1,
            "result": {"tools": [{"name": t["name"],
                                  "description": t.get("description", ""),
                                  "inputSchema": t["inputSchema"]} for t in tools]}
        }
        task = (f"{target.get('description') or target['name']} "
                f"Discover what you can do first, then do it.")
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
            {"role": "assistant", "content": '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'},
            {"role": "tool", "content": json.dumps(listing)},
        ]
        batch.append((msgs, tools, task, listing))
    return batch


def _extract_calls(text: str) -> List[Dict[str, Any]]:
    """Brace-matched JSON-RPC extraction from generation text."""
    out = []
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        depth, in_str, esc = 0, False, False
        for j in range(i, len(text)):
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[i:j + 1])
                        if isinstance(obj, dict) and "method" in obj:
                            out.append(obj)
                    except Exception:
                        pass
                    break
    return out


def _validate_tool_call(call: Dict, tools_by_name: Dict) -> tuple:
    """Check a JSON-RPC tools/call against the real schema. Returns (ok, reason)."""
    if not isinstance(call, dict):
        return False, "not an object"
    if call.get("method") != "tools/call":
        return False, f"method {call.get('method')!r}"

    params = call.get("params") or {}
    name = params.get("name")
    if name not in tools_by_name:
        return False, f"unknown tool {name!r}"

    schema = tools_by_name[name]["inputSchema"]
    args = params.get("arguments")
    if not isinstance(args, dict):
        return False, "arguments not an object"

    props = schema.get("properties") or {}
    for req in schema.get("required") or []:
        if req not in args:
            return False, f"missing required arg {req!r}"
    for k, v in args.items():
        if k not in props:
            return False, f"unknown arg {k!r} for {name}"
        spec = props[k]
        want = spec.get("type")
        if want == "string" and not isinstance(v, str):
            return False, f"{k} should be string"
        if want == "integer" and not isinstance(v, int) or isinstance(v, bool):
            return False, f"{k} should be integer"
        if want == "number" and not isinstance(v, (int, float)) or isinstance(v, bool):
            return False, f"{k} should be number"
        if want == "boolean" and not isinstance(v, bool):
            return False, f"{k} should be boolean"
        if want == "array" and not isinstance(v, list):
            return False, f"{k} should be array"
        if want in ("integer", "number") and isinstance(v, (int, float)) and not isinstance(v, bool):
            if spec.get("minimum") is not None and v < spec["minimum"]:
                return False, f"{k} below minimum"
            if spec.get("maximum") is not None and v > spec["maximum"]:
                return False, f"{k} above maximum"
        if spec.get("enum") is not None and v not in spec["enum"]:
            return False, f"{k}={v!r} not in enum"
        if spec.get("pattern") and isinstance(v, str):
            import re as _re
            if not _re.search(spec["pattern"], v):
                return False, f"{k} fails pattern"
    return True, "ok"


# ---------------------------------------------------------------------------
# Teacher backends
# ---------------------------------------------------------------------------

class _HFTeacher:
    """transformers + bitsandbytes nf4 teacher."""
    batch_size: int

    def __init__(self, config):
        self.batch_size = config.teacher_batch
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError("bitsandbytes is required for the HF teacher")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=None,  # will be set after torch import
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        import torch
        quant_config.bnb_4bit_compute_dtype = torch.bfloat16

        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            config.teacher_model,
            quantization_config=quant_config,
            device_map={"": 0},
            max_memory={0: config.teacher_max_memory},
            cache_dir=config.hf_home,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.teacher_model, cache_dir=config.hf_home)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def complete(self, prompts):
        import torch
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=224,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        cut = inputs.input_ids.shape[1]
        return [self.tokenizer.decode(o[cut:], skip_special_tokens=True) for o in outputs]

    def close(self):
        import torch
        del self.model
        torch.cuda.empty_cache()


class _VLLMTeacher:
    """vLLM teacher — continuous batching + paged KV."""
    batch_size: int

    def __init__(self, config):
        from vllm import LLM, SamplingParams
        # AutoTokenizer was imported in _HFTeacher.__init__ and used here,
        # where nothing binds it - a NameError waiting for the first recipe
        # with a synth expert on the vLLM path. transformers cannot move to
        # module level (that would put torch behind `ms-moe-maker validate`),
        # so it is imported here, in the function that uses it.
        from transformers import AutoTokenizer
        self._SamplingParams = SamplingParams
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.teacher_model, cache_dir=config.hf_home)
        self.llm = LLM(
            model=config.teacher_model,
            dtype="bfloat16",
            gpu_memory_utilization=config.vllm_gpu_util,
            max_model_len=config.vllm_max_len,
            enable_prefix_caching=True,
            cache_dir=config.hf_home,
            trust_remote_code=True,
        )
        if config.vllm_quantization:
            self.llm = LLM(
                model=config.teacher_model,
                dtype="bfloat16",
                gpu_memory_utilization=config.vllm_gpu_util,
                max_model_len=config.vllm_max_len,
                enable_prefix_caching=True,
                quantization=config.vllm_quantization,
                cache_dir=config.hf_home,
                trust_remote_code=True,
            )
        self.params = SamplingParams(
            temperature=0.7, top_p=0.8, top_k=20,
            repetition_penalty=1.05,
            max_tokens=224,
        )
        self.batch_size = config.vllm_batch

    def complete(self, prompts):
        outs = self.llm.generate(prompts, self.params)
        return [o.outputs[0].text for o in outs]

    def close(self):
        del self.llm
        import torch
        torch.cuda.empty_cache()
