import sys
import importlib

sys.path.insert(0, "D:/serenDaemon/SerenCore/MsMoEMaker/MsMoEMaker")

modules = [
    "ms_moe_maker.config",
    "ms_moe_maker.data",
    "ms_moe_maker.finetune",
    "ms_moe_maker.stitch",
    "ms_moe_maker.router",
    "ms_moe_maker.export",
    "ms_moe_maker.builder",
    "ms_moe_maker.runner",
    "ms_moe_maker.levers",
    "ms_moe_maker.manifest",
    "ms_moe_maker.events",
    "ms_moe_maker.evalrecord",
    "ms_moe_maker.stages",
    "ms_moe_maker.corpus",
    "ms_moe_maker.validators",
]

ok = 0
fail = 0
for m in modules:
    try:
        importlib.import_module(m)
        print(f"OK   {m}")
        ok += 1
    except Exception as e:
        print(f"FAIL {m}: {type(e).__name__}: {e}")
        fail += 1

print(f"\n{ok} OK, {fail} FAIL")
