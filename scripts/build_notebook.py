"""Generate notebooks/colab_pipeline.ipynb (plain nbformat-4 JSON, no dependencies)."""

import json
from pathlib import Path

CELLS = []


def md(s):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md("""
# Lost in Transcription — Spanish–English: end-to-end Colab pipeline

**Data → LoRA fine-tune Whisper → merge → CTranslate2 → `submission.zip` → local check.**
Run the cells top to bottom. Every stage is resumable: checkpoints and outputs live on Google Drive.

**Before you start**
1. Runtime → *Change runtime type* → GPU (T4 free tier works; A100/L4 = faster/better).
2. **Data is uploaded once**: the data cell asks you to pick the two archives (`…miami.tar.gz`, `…enspa_dev.tar.gz`) from your
   computer and saves them to `MyDrive/lit_data/` (any subfolder of it is also found); every later run reads them from Drive (prepared clips are cached there too).
   They are never sent anywhere else.
3. **Private repo**: create a GitHub token (Settings → Developer settings → Fine-grained tokens → this repo, *Contents: read*) and add it as
   a Colab secret named `GITHUB_TOKEN` (key icon in the left bar, enable *Notebook access*).
4. Set `SMOKE = True` first (tiny model, ~5 min) to prove the whole chain works, then set it `False`.
""")

code('''
# ---- settings ---------------------------------------------------------------------------------
SMOKE      = True                     # True: tiny model + 3 conversations, proves the pipeline. False: real run
PRESET     = "auto"                   # "t4" | "a100" | "auto" (pick by GPU)
RUN_NAME   = "run1"                   # a new name = a fresh run; the same name = resume
REPO_URL   = "https://github.com/raut7218/lit-spanglish-asr.git"   # private repo: add a Colab secret GITHUB_TOKEN (see below)
DRIVE_DATA = "MyDrive/lit_data"       # folder in Drive holding the two .tar.gz files
DRIVE_RUNS = "MyDrive/lit_runs"       # checkpoints / exports are written here
LANGUAGE   = "es"                     # decoder language token ("es" or "en"); compare both on dev
ZERO_SHOT_BASELINE = True             # score the un-tuned base model on dev first (sanity + reference)
MAX_TRAIN_MINUTES  = 0                # e.g. 210 stops training cleanly before a Colab session limit; re-run to resume
''')

code('''
# ---- environment ------------------------------------------------------------------------------
import os, sys, json, glob, shutil, subprocess, time, textwrap
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")
if IN_COLAB and "google.colab" in sys.modules:
    from google.colab import drive
    drive.mount("/content/drive")
    WORK = Path("/content")
    DRIVE = Path("/content/drive")
else:  # local dry run of this notebook (used by CI/smoke)
    WORK = Path(os.environ.get("LIT_WORK", "/tmp/lit_notebook_work")); WORK.mkdir(parents=True, exist_ok=True)
    DRIVE = Path(os.environ.get("LIT_FAKE_DRIVE", str(WORK / "drive")))

def sh(cmd, check=True, env=None, cwd=None):
    """Run a shell command, streaming output live."""
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=cwd, executable="/bin/bash")
    for line in p.stdout:
        print(line, end="", flush=True)
    rc = p.wait()
    if check and rc != 0:
        raise RuntimeError(f"command failed ({rc}): {cmd}")
    return rc

sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo 'NO GPU'", check=False)
import torch
GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
print("GPU:", GPU)
if PRESET == "auto":
    PRESET = "a100" if GPU and any(k in GPU for k in ("A100", "L4", "H100", "A10")) else "t4"
CONFIG = "configs/smoke.yaml" if SMOKE else f"configs/colab_{PRESET}.yaml"
print("preset:", PRESET, "| config:", CONFIG, "| smoke:", SMOKE)
''')

code('''
# ---- code + dependencies ----------------------------------------------------------------------
REPO = Path(os.environ.get("LIT_REPO", WORK / "lit-spanglish-asr"))
if not REPO.exists():
    token = None
    try:
        from google.colab import userdata; token = userdata.get("GITHUB_TOKEN")   # only needed for a private repo
    except Exception:
        pass
    url = REPO_URL.replace("https://", f"https://{token}@") if token else REPO_URL
    sh(f"git clone --depth 1 {url} {REPO}")
else:
    sh("git pull --ff-only || true", check=False, cwd=REPO)

if IN_COLAB and "google.colab" in sys.modules:
    # versions pinned to the competition runtime (transformers 4.57.6 / peft 0.20 / faster-whisper 1.2.1)
    sh('pip install -q "transformers==4.57.6" "peft==0.20.0" "faster-whisper==1.2.1" "ctranslate2==4.8.2" jiwer typer pyyaml soundfile sentencepiece pytest accelerate')
    sh("ffmpeg -hide_banner -encoders 2>/dev/null | grep -E 'libopus|libmp3lame|amr' || (apt-get -qq install -y ffmpeg)", check=False)
ENV = dict(os.environ, PYTHONPATH=str(REPO / "src"), TOKENIZERS_PARALLELISM="false")
PY = sys.executable
sh(f"{PY} -m pytest -q -x tests", cwd=REPO, env=ENV)     # unit tests: scorer parity, CHAT parser, segmentation, casing, augmentation
''')

code('''
# ---- data: kept on YOUR DRIVE so you upload it only once -----------------------------------------
RAW, PREP = WORK / "raw", WORK / "data_prepared"
src = DRIVE / DRIVE_DATA
src.mkdir(parents=True, exist_ok=True)
def find_tars():
    """Search DRIVE_DATA recursively, so it works whether the archives sit in lit_data/ or lit_data/data/."""
    hit = lambda pat: next(iter(sorted(glob.glob(str(src / "**" / pat), recursive=True))), None)
    return hit("*miami*.tar.gz"), hit("*enspa_dev*.tar.gz")
miami_tar, dev_tar = find_tars()
if not (miami_tar and dev_tar):
    print(f"Archives not on Drive yet ({src}). One-time upload: choose BOTH .tar.gz files from your computer "
          "(miami + enspa_dev). They are saved to your Drive, so later runs skip this step.\\n"
          "(If the browser upload is slow/fails, drag the two files into Drive > lit_data/ (any subfolder) in another tab instead, then re-run this cell.)")
    if IN_COLAB and "google.colab" in sys.modules:
        from google.colab import files
        up = files.upload()                       # opens a file picker on your computer
        for name in list(up):
            shutil.move(name, src / name)
        del up
        miami_tar, dev_tar = find_tars()
assert miami_tar and dev_tar, f"put *miami*.tar.gz and *enspa_dev*.tar.gz in {src}"
cache = DRIVE / DRIVE_RUNS / f"prepared_{'smoke' if SMOKE else 'full'}.tar"

if not (PREP / "train.jsonl").exists():
    if cache.exists():                       # fast restart after a session reset
        sh(f"mkdir -p {PREP} && tar xf {cache} -C {PREP}")
    else:
        RAW.mkdir(exist_ok=True)
        if not (RAW / "miami").exists(): sh(f"tar xzf {miami_tar} -C {RAW}")
        if not (RAW / "enspa_dev").exists(): sh(f"tar xzf {dev_tar} -C {RAW}")
        extra = "--max_convs 3 --holdout_hours 0.05" if SMOKE else "--holdout_hours 1.5"
        sh(f"{PY} -m lit.prepare_data --miami_dir {RAW/'miami'} --dev_dir {RAW/'enspa_dev'} --out_dir {PREP} --workers 2 {extra}", cwd=REPO, env=ENV)
        cache.parent.mkdir(parents=True, exist_ok=True)
        sh(f"tar cf {cache} -C {PREP} .")    # cache prepared FLAC clips on Drive
for name in ("train", "miami_holdout", "dev"):
    rows = [json.loads(l) for l in open(PREP / f"{name}.jsonl", encoding="utf-8")]
    print(f"{name:14s} {len(rows):6d} clips  {sum(r['duration'] for r in rows)/3600:6.2f} h")
print("example target:", json.loads(open(PREP/'train.jsonl').readline())["text"])
''')

code('''
# ---- zero-shot baseline (un-tuned base model on the honest dev set) ----------------------------
import yaml
cfg0 = yaml.safe_load(open(REPO / CONFIG))
BASE = cfg0["model"]
if ZERO_SHOT_BASELINE:
    for lang in ("es", "en"):
        print(f"--- zero-shot {BASE}, language token = {lang}")
        sh(f"{PY} -m lit.evaluate --model {BASE} --data_dir {PREP} --language {lang} --beams 1 --max_clips {12 if SMOKE else 0}", cwd=REPO, env=ENV)
''')

code('''
# ---- train (LoRA) — resumable: just re-run this cell after a disconnect -------------------------
RUN = DRIVE / DRIVE_RUNS / (RUN_NAME + ("_smoke" if SMOKE else ""))
sets = f"data_dir={PREP} out_dir={RUN} language={LANGUAGE} max_train_minutes={MAX_TRAIN_MINUTES}"
sh(f"{PY} -u -m lit.train --config {CONFIG} --set {sets}", cwd=REPO, env=ENV)
print(open(RUN / "final_adapter" / "final.json").read()[:600] if (RUN / "final_adapter" / "final.json").exists() else "training not finished yet - re-run this cell to resume")
''')

code('''
# ---- evaluate the final adapter (dev = honest, Miami hold-out = speaker-disjoint) ---------------
assert (RUN / "final_adapter").exists(), "training has not finished"
for split in ("dev", "miami_holdout"):
    sh(f"{PY} -m lit.evaluate --model {BASE} --adapter {RUN/'final_adapter'} --data_dir {PREP} --split {split} --language {LANGUAGE} --beams {1 if SMOKE else 4} --max_clips {12 if SMOKE else 0}", cwd=REPO, env=ENV)
''')

code('''
# ---- export: merge LoRA -> CTranslate2 (faster-whisper) -> submission.zip ------------------------
EXPORT = DRIVE / DRIVE_RUNS / (RUN_NAME + ("_smoke" if SMOKE else "")) / "export"
quant = "float32" if SMOKE else "float16"
sh(f"{PY} -m lit.export --model {BASE} --adapter {RUN/'final_adapter'} --data_dir {PREP} --out {EXPORT} --quantization {quant}", cwd=REPO, env=ENV)
ZIP = RUN / "submission.zip"
cfg_json = json.dumps({"language": LANGUAGE, "beam_size": 2 if SMOKE else 5, "compute_type": "float16"})
sh(f"{PY} scripts/make_submission.py --export {EXPORT} --out {ZIP} --cfg '{cfg_json}'", cwd=REPO, env=ENV)
''')

code('''
# ---- verify the zip end-to-end like the platform does (unzip -> main.py -> submission.csv -> WER) -
sh(f"{PY} scripts/run_submission_local.py --zip {ZIP} --prepared {PREP} --dev_raw {RAW/'enspa_dev'} --n {12 if SMOKE else 0}", cwd=REPO, env=ENV)
print("\\nsubmission.zip:", ZIP, f"({ZIP.stat().st_size/1e9:.2f} GB)")
''')

code('''
# ---- download the zip (upload it on the competition "Submissions" page; run the platform smoke test first) --
if IN_COLAB and "google.colab" in sys.modules:
    from google.colab import files
    files.download(str(ZIP))     # it is also saved on Drive next to the checkpoints
''')

md("""
### After the first full run
* Compare dev WER for `LANGUAGE="es"` vs `"en"` (re-run the evaluation cell) and use the better one.
* Tune only what moves dev WER: learning rate (`1e-4`→`2e-4`), LoRA rank, `epochs`, augmentation probabilities in the config.
* **Last step only:** retrain once with `include_dev_in_train=true` (and `--include_dev_in_lexicon` in export) using the
  hyper-parameters you already selected. Keep the previous zip as the known-good fallback.
* Check the platform's *Code submission format* page for zip-size and time limits; use `quantization=int8_float16` in the
  export cell to halve the model size if needed.
""")

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                  "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parents[1] / "notebooks" / "colab_pipeline.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, len(CELLS), "cells")
