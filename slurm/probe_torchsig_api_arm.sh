#!/bin/bash
# probe_torchsig_api_arm.sh — Deep introspection of the torchsig 2.x wideband API,
# to write the real WBSig53 loader. Reuses the venv from probe_torchsig_arm.sh
# ($WORK/envs/rfbench-arm-detection). CPU-only, fast (no install).
#
# Usage: sbatch slurm/probe_torchsig_api_arm.sh
#SBATCH --job-name=rfbench_tsig_api
#SBATCH --output=/lustre/work/pdl16831/udl79f933/logs/rfbench_tsig_api_%j.out
#SBATCH --error=/lustre/work/pdl16831/udl79f933/logs/rfbench_tsig_api_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
# cluster mono-partition (defq*, GB200/ARM uniquement) -- pas de contrainte d'architecture requise
# (confirmé via `sinfo -o "%P %f %c %G"`, seule feature reportée: location=local)

set -uo pipefail
WORK=/lustre/work/pdl16831/udl79f933
VENV="$WORK/envs/rfbench-arm-detection"
echo "=== node=$(hostname) arch=$(uname -m) date=$(date -Is) ==="

"$VENV/bin/python" - <<'PY'
import inspect, pkgutil, importlib, traceback

def show(title): print(f"\n===== {title} =====")

import torchsig
show("torchsig version + top-level submodules")
print("version", getattr(torchsig, "__version__", "?"))
print([x.name for x in pkgutil.iter_modules(torchsig.__path__)])

show("torchsig.datasets.datasets public names")
try:
    from torchsig.datasets import datasets as D
    print([n for n in dir(D) if not n.startswith("_")])
except Exception:
    traceback.print_exc()

show("torchsig.datasets.default_configs")
try:
    from torchsig.datasets import default_configs as C
    names = [n for n in dir(C) if not n.startswith("_")]
    print(names)
    for n in names:
        obj = getattr(C, n)
        if any(k in n.lower() for k in ("wide", "detect")):
            print(f"  -- {n}: {type(obj)}")
except Exception:
    traceback.print_exc()

show("Wideband dataset class discovery (scan all torchsig submodules)")
found = []
for mod in pkgutil.walk_packages(torchsig.__path__, prefix="torchsig."):
    name = mod.name
    if any(k in name.lower() for k in ("wide", "detect")):
        found.append(name)
print("modules matching wide/detect:", found)
for mn in found:
    try:
        m = importlib.import_module(mn)
        classes = [n for n in dir(m) if not n.startswith("_") and n[0].isupper()]
        print(f"  [{mn}] classes: {classes[:25]}")
    except Exception as e:
        print(f"  [{mn}] import failed: {e}")

show("Signatures of likely Wideband dataset builders")
try:
    from torchsig.datasets import datasets as D
    for n in dir(D):
        if n.lower().startswith("wideband") or "wideband" in n.lower():
            obj = getattr(D, n)
            if inspect.isclass(obj):
                try:
                    print(f"  {n}{inspect.signature(obj.__init__)}")
                except (TypeError, ValueError):
                    print(f"  {n}: <no signature>")
                doc = (inspect.getdoc(obj) or "").splitlines()[:6]
                print("   doc:", " / ".join(doc))
except Exception:
    traceback.print_exc()

show("Try to build a tiny wideband dataset + dump ONE sample structure")
try:
    from torchsig.datasets.datasets import NewWideband  # torchsig 2.x name (guess A)
    builder = NewWideband
except Exception:
    builder = None
if builder is None:
    try:
        from torchsig.datasets.datasets import Wideband  # guess B
        builder = Wideband
    except Exception:
        builder = None
print("resolved builder:", builder)
if builder is not None:
    try:
        sig = inspect.signature(builder.__init__)
        print("builder init:", sig)
    except Exception:
        pass
PY
rc=$?
echo "=================================================="
[ "$rc" -eq 0 ] && echo "RESULT: SUCCESS — torchsig 2.x API mapped (see above)" || echo "RESULT: PROBE rc=$rc"
exit "$rc"
