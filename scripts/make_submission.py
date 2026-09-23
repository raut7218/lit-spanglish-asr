"""Assemble submission.zip (main.py at the archive root) from an export dir.

    python scripts/make_submission.py --export EXPORT --out submission.zip [--cfg '{"beam_size": 5}']
"""

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIT_FILES = ["__init__.py", "normalize.py", "casing.py", "postprocess.py", "rules.py", "infer.py", "canary.py", "audio.py"]
MODEL_FILES = ["config.json", "model.safetensors", "tokenizer.model"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="final_model dir from lit.train (config.json, model.safetensors, tokenizer.model)")
    ap.add_argument("--out", default="submission.zip")
    ap.add_argument("--cfg", default="{}", help="JSON overrides for infer_config.json (language, beam_size, ...)")
    ap.add_argument("--cfg_file", help="JSON file with the overrides (wins over --cfg)")
    a = ap.parse_args()

    exp = Path(a.export)
    missing = [f for f in MODEL_FILES if not (exp / f).exists()]
    assert not missing, f"{exp} lacks {missing} - point --export at lit.train's final_model dir"
    stage = Path(tempfile.mkdtemp())
    shutil.copy(ROOT / "submission_src" / "main.py", stage / "main.py")
    (stage / "lit").mkdir()
    for f in LIT_FILES:
        shutil.copy(ROOT / "src" / "lit" / f, stage / "lit" / f)
    (stage / "model").mkdir()
    for f in MODEL_FILES:
        shutil.copy(exp / f, stage / "model" / f)
    (stage / "model" / "infer_config.json").write_text(json.dumps(json.loads(Path(a.cfg_file).read_text()) if a.cfg_file else json.loads(a.cfg), indent=2))

    out = Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    # Build on LOCAL disk (zipfile seeks back to patch headers, which network/FUSE mounts such as Colab's
    # Drive handle badly), then copy sequentially to the destination and verify the copy.
    local = Path(tempfile.mkdtemp()) / "submission.zip"
    with zipfile.ZipFile(local, "w") as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                rel = p.relative_to(stage).as_posix()
                z.write(p, rel, compress_type=zipfile.ZIP_STORED if p.suffix == ".safetensors" else zipfile.ZIP_DEFLATED)
    shutil.rmtree(stage)
    if out.exists():
        out.unlink()
    shutil.copyfile(local, out)
    os.sync()
    assert out.stat().st_size == local.stat().st_size, "copied zip has a different size"
    with zipfile.ZipFile(out) as z:  # reads the central directory back from the destination
        names = z.namelist()
        assert "main.py" in names, "main.py must be at the zip root"
    shutil.rmtree(local.parent, ignore_errors=True)
    print(f"[make_submission] {out} ({out.stat().st_size/1e9:.2f} GB, {len(names)} files, main.py at root)")


if __name__ == "__main__":
    main()
