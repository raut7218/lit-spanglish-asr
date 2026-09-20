"""Assemble submission.zip (main.py at the archive root) from an export dir.

    python scripts/make_submission.py --export EXPORT --out submission.zip [--cfg '{"beam_size": 5}']
"""

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIT_FILES = ["__init__.py", "normalize.py", "casing.py", "postprocess.py", "rules.py", "infer.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="dir produced by lit.export (has ct2/ + casing_lexicon.json)")
    ap.add_argument("--out", default="submission.zip")
    ap.add_argument("--cfg", default="{}", help="JSON overrides for infer_config.json (language, beam_size, ...)")
    ap.add_argument("--cfg_file", help="JSON file with the overrides (wins over --cfg)")
    ap.add_argument("--lexicon", action="store_true", help="ship casing_lexicon.json (off by default: it hurt dev WER)")
    a = ap.parse_args()

    exp = Path(a.export)
    assert (exp / "ct2" / "model.bin").exists(), f"{exp}/ct2/model.bin missing - run lit.export first"
    stage = Path(tempfile.mkdtemp())
    shutil.copy(ROOT / "submission_src" / "main.py", stage / "main.py")
    (stage / "lit").mkdir()
    for f in LIT_FILES:
        shutil.copy(ROOT / "src" / "lit" / f, stage / "lit" / f)
    (stage / "model").mkdir()
    shutil.copytree(exp / "ct2", stage / "model" / "ct2")
    if a.lexicon and (exp / "casing_lexicon.json").exists():
        shutil.copy(exp / "casing_lexicon.json", stage / "model" / "casing_lexicon.json")
    (stage / "model" / "infer_config.json").write_text(json.dumps(json.loads(Path(a.cfg_file).read_text()) if a.cfg_file else json.loads(a.cfg), indent=2))

    out = Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w") as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                rel = p.relative_to(stage).as_posix()
                z.write(p, rel, compress_type=zipfile.ZIP_STORED if p.suffix == ".bin" else zipfile.ZIP_DEFLATED)
    shutil.rmtree(stage)
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        assert "main.py" in names, "main.py must be at the zip root"
    print(f"[make_submission] {out} ({out.stat().st_size/1e9:.2f} GB, {len(names)} files, main.py at root)")


if __name__ == "__main__":
    main()
