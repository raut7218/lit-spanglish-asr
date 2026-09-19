"""Execute the notebook's code cells in order on this machine (no Colab): dry-run of the whole pipeline.

    LIT_FAKE_DRIVE=/path/with/MyDrive/lit_data  python scripts/run_notebook_local.py
Cells run in one shared namespace; the notebook's own IN_COLAB switch keeps Colab-only calls disabled.
"""
import json
import sys
from pathlib import Path

nb = json.load(open(Path(__file__).resolve().parents[1] / "notebooks" / "colab_pipeline.ipynb"))
ns = {"__name__": "__notebook__"}
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    print(f"\n########## cell {i} ##########", flush=True)
    exec(compile("".join(c["source"]), f"cell{i}", "exec"), ns)
print("\nNOTEBOOK RAN TO COMPLETION")
