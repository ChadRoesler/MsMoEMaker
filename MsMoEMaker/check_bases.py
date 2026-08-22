"""Are every size's checkpoints actually reachable from THIS box?

preflight checks only the base for the size you are building, which is right -
but it means a 3B rung can die on a repo you could have checked in ten seconds
before you started. Run this once after a token change or an HF outage.

    python check_bases.py            # anonymous + whatever token you have
    HF_TOKEN= python check_bases.py  # force anonymous, to isolate the token
"""
import os
import sys

from huggingface_hub import model_info
from huggingface_hub.utils import (GatedRepoError, HfHubHTTPError,
                                   RepositoryNotFoundError)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ms_moe_maker.config import MODEL_SIZES  # noqa: E402

tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
print(f"token present: {bool(tok)}\n")

bad = 0
for size, (safe, ablated) in MODEL_SIZES.items():
    for role, repo in (("safe", safe), ("ablated", ablated)):
        if not repo:
            continue
        try:
            info = model_info(repo)
            print(f"  ok    {size:>5} {role:<8} {repo}  ({info.downloads or 0:,} dl)")
        except GatedRepoError:
            bad += 1
            print(f"  GATED {size:>5} {role:<8} {repo}")
            print(f"        -> accept the terms on the model page, then `hf auth login`")
        except RepositoryNotFoundError:
            bad += 1
            print(f"  404   {size:>5} {role:<8} {repo}")
            # THE MISLEADING-ERROR TRAP. huggingface_hub raises
            # RepositoryNotFoundError for "does not exist" AND for "you are not
            # allowed to see it" - a bad or expired token looks exactly like a
            # deleted model. Check the token before you edit the table.
            print(f"        -> this is ALSO what an invalid token looks like. "
                  f"Try: HF_TOKEN= python {os.path.basename(__file__)}")
        except HfHubHTTPError as exc:
            bad += 1
            print(f"  ERR   {size:>5} {role:<8} {repo}: {exc.response.status_code}")

print(f"\n{bad} unreachable" if bad else "\nall reachable")
sys.exit(1 if bad else 0)
