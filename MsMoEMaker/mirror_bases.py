"""Mirror the base models this tool depends on, so a deleted repo cannot end a project.

THE FAILURE THIS EXISTS FOR, and it already happened: the entire
`huihui-ai/Qwen2.5-Coder-*-Instruct-abliterated` family was deleted upstream.
It survives only as third-party GGUF/AWQ mirrors. A build that had worked for
months died at preflight with a 404, and the 404 read exactly like a bad
credential - `huggingface_hub` raises RepositoryNotFoundError for "gone" AND
for "not allowed", so a deleted model and an expired token are the same
exception.

You cannot stop somebody deleting a repo. You CAN stop that from being your
problem: mirror what you depend on, and point `models:` at the mirror.

    python mirror_bases.py --to /mnt/nvme/basemodels
    python mirror_bases.py --to /mnt/nvme/basemodels --sizes 0.5B 1.5B
    python mirror_bases.py --to /mnt/nvme/basemodels --push-to goblinModeMan

It prints a ready-to-paste `models:` block for ~/.msmoe/defaults.yaml when it
finishes, because a mirror nothing points at is just a big directory.

Needs only huggingface_hub - no torch, so this runs anywhere `validate` does.
"""
from __future__ import annotations

import argparse
import os
import sys

# Weights and the metadata a trainer needs. NOT the quantised formats: a GGUF
# is an output of this pipeline, not an input to it, and pulling every variant
# turns a 1 GB mirror into 20.
ALLOW = ["*.safetensors", "*.json", "*.txt", "*.model", "*.py", "*.md"]
IGNORE = ["*.gguf", "*.onnx", "*.bin", "*.pth", "*.msgpack", "*.h5"]


def _sizes_table():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ms_moe_maker.config import MODEL_SIZES
    return MODEL_SIZES


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--to", required=True,
                    help="local directory to mirror into")
    ap.add_argument("--sizes", nargs="*", default=None,
                    help="sizes to mirror (default: every size in the table)")
    ap.add_argument("--half", choices=["safe", "ablated", "both"],
                    default="both",
                    help="which half of each entry to mirror")
    ap.add_argument("--push-to", metavar="ORG", default=None,
                    help="also upload each mirror to this HF namespace "
                         "(private). Needs a token with write scope.")
    ap.add_argument("--public", action="store_true",
                    help="with --push-to, create PUBLIC repos instead of private")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    table = _sizes_table()
    sizes = args.sizes or list(table)
    unknown = [s for s in sizes if s not in table]
    if unknown:
        print(f"unknown size(s) {unknown}. Known: {', '.join(table)}",
              file=sys.stderr)
        return 1

    os.makedirs(args.to, exist_ok=True)
    mirrored = {}
    failed = []

    for size in sizes:
        safe, ablated = table[size]
        wanted = []
        if args.half in ("safe", "both") and safe:
            wanted.append(("safe", safe))
        if args.half in ("ablated", "both") and ablated and ablated != safe:
            wanted.append(("ablated", ablated))

        for role, repo in wanted:
            dest = os.path.join(args.to, repo.replace("/", "__"))
            print(f"\n=== {size} {role}: {repo}")
            try:
                path = snapshot_download(
                    repo_id=repo, local_dir=dest,
                    allow_patterns=ALLOW, ignore_patterns=IGNORE)
            except Exception as exc:  # noqa: BLE001 - report, do not stop
                # ONE DEAD REPO MUST NOT END THE SWEEP. The whole point is to
                # find out what is still there; aborting on the first miss
                # tells you the least useful thing at the highest cost.
                print(f"  FAILED: {type(exc).__name__}: {exc}"[:400])
                failed.append((size, role, repo))
                continue
            print(f"  -> {path}")
            mirrored[(size, role)] = path

            if args.push_to:
                target = f"{args.push_to}/{repo.split('/')[-1]}"
                try:
                    from huggingface_hub import HfApi
                    api = HfApi()
                    api.create_repo(target, private=not args.public,
                                    exist_ok=True)
                    api.upload_folder(folder_path=path, repo_id=target,
                                      commit_message=f"mirror of {repo}")
                    print(f"  -> pushed to {target}"
                          f"{'' if args.public else '  (private)'}")
                    mirrored[(size, role)] = target
                except Exception as exc:  # noqa: BLE001
                    print(f"  PUSH FAILED: {type(exc).__name__}: {exc}"[:400])

    # ── the part that makes the mirror load-bearing ────────────────────────
    #
    # A mirror nothing points at is just a big directory. `models:` is a
    # BOX-ONLY defaults block (a recipe may not set it), which is exactly right
    # here: where your weights live is a fact about your machine, not about the
    # build, and every recipe on the box should inherit it without saying so.
    print("\n" + "=" * 70)
    if not mirrored:
        print("nothing mirrored.")
    else:
        print("Paste into ~/.msmoe/defaults.yaml:\n")
        print("models:")
        for size in sizes:
            s = mirrored.get((size, "safe"))
            a = mirrored.get((size, "ablated"))
            if not (s or a):
                continue
            print(f'  "{size}":')
            if s:
                print(f"    safe: {s}")
            if a:
                print(f"    abliterated: {a}")
        print("\nThen: ms-moe-maker validate <recipe>   # provenance names this file")

    if failed:
        print("\nUNREACHABLE (these are the ones that will kill a build later):")
        for size, role, repo in failed:
            print(f"  {size:>5} {role:<8} {repo}")
        print("\nA 404 here is ALSO what an invalid token looks like. If every "
              "repo failed, check `hf auth whoami` before you edit the table.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
