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
LIT_FILES = ["__init__.py", "normalize.py", "casing.py", "postprocess.py", "rules.py", "infer.py", "mbr.py", "qwen.py", "audio.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="dir produced by lit.export (has ct2/ + casing_lexicon.json)")
    ap.add_argument("--out", default="submission.zip")
    ap.add_argument("--cfg", default="{}", help="JSON overrides for infer_config.json (language, beam_size, ...)")
    ap.add_argument("--cfg_file", help="JSON file with the overrides (wins over --cfg)")
    ap.add_argument("--lexicon", action="store_true", help="ship casing_lexicon.json (off by default: it hurt dev WER)")
    ap.add_argument("--extra", nargs="*", default=[], help="more exports for an MBR ensemble: model/ct2_2, ct2_3, ... "
                    "(infer_config.json gets systems=[ct2, ct2_2, ...])")
    ap.add_argument("--qwen", nargs="*", default=[], help="merged Qwen3-ASR dirs (lit.train_qwen out/merged): model/qwen, "
                    "qwen_2, ...; they go FIRST in systems (MBR ties -> first) unless the cfg sets `systems`")
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
    systems = ["ct2"]
    for k, e in enumerate(a.extra, 2):
        assert (Path(e) / "ct2" / "model.bin").exists(), f"{e}/ct2/model.bin missing"
        shutil.copytree(Path(e) / "ct2", stage / "model" / f"ct2_{k}")
        systems.append(f"ct2_{k}")
    qsys = []
    for k, q in enumerate(a.qwen, 1):
        assert (Path(q) / "config.json").exists() and list(Path(q).glob("*.safetensors")), f"{q}: not a merged Qwen dir"
        name = "qwen" if k == 1 else f"qwen_{k}"
        shutil.copytree(q, stage / "model" / name)
        qsys.append(name)
    systems = qsys + systems
    if a.lexicon and (exp / "casing_lexicon.json").exists():
        shutil.copy(exp / "casing_lexicon.json", stage / "model" / "casing_lexicon.json")
    icfg = json.loads(Path(a.cfg_file).read_text()) if a.cfg_file else json.loads(a.cfg)
    if "systems" in icfg:
        assert set(icfg["systems"]) <= set(systems), f"cfg systems {icfg['systems']} not all shipped ({systems})"
    elif len(systems) > 1 or qsys:
        icfg["systems"] = systems
    (stage / "model" / "infer_config.json").write_text(json.dumps(icfg, indent=2))

    out = Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    # Build on LOCAL disk (zipfile seeks back to patch headers, which network/FUSE mounts such as Colab's
    # Drive handle badly), then copy sequentially to the destination and verify the copy.
    local = Path(tempfile.mkdtemp()) / "submission.zip"
    with zipfile.ZipFile(local, "w") as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                rel = p.relative_to(stage).as_posix()
                z.write(p, rel, compress_type=zipfile.ZIP_STORED if p.suffix in (".bin", ".safetensors") else zipfile.ZIP_DEFLATED)
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
