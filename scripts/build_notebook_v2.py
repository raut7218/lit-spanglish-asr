"""Generate notebooks/colab_pipeline.ipynb (v2 recipe: tests -> data -> baseline -> train -> analyse -> zip -> validate)."""

import json
from pathlib import Path

CELLS = []


def md(s):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md("""
# Lost in Transcription — v2 pipeline (Colab A100)

Tests → data (conventions, conversation windows, non-speech, codec bank) → baseline of the old model on the new
validation sets → **v2 training** (frozen lower encoder, GPU augmentation, EMA, language-matched sampler) →
error analysis → rules chosen on dev **and** hold-out → `submission.zip` → validation → optional download.
Everything is resumable from Google Drive. Set `DOWNLOAD_ZIP = True` only at the very end.
""")

code('''
RUN_NAME    = "run2"
REPO_URL    = "https://github.com/raut7218/lit-spanglish-asr.git"
DRIVE_DATA  = "MyDrive/lit_data"
DRIVE_RUNS  = "MyDrive/lit_runs"
CONFIG      = "configs/colab_a100_v2.yaml"
EXTRA_SET   = ""             # e.g. "max_steps=1500 lora.r=32 dev_in_train=spk1"
DEV_IN_TRAIN = "none"        # none | spk1 | spk2 | all   (all = final fit; dev WER is then contaminated)
BUILD_BANK  = True           # pre-render Opus->MP3 codec variants (~15 min once, cached on Drive)
RUN_E2E_TEST = True          # tiny end-to-end training test on the GPU
BASELINE_ADAPTER = "run1/final_adapter"   # old model, scored on the NEW validation sets for a fair comparison
DOWNLOAD_ZIP = False
MAX_TRAIN_MINUTES = 0
''')

code('''
import os, sys, json, glob, shutil, subprocess, time
from pathlib import Path
IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")
if IN_COLAB and "google.colab" in sys.modules:
    from google.colab import drive
    drive.mount("/content/drive"); WORK = Path("/content"); DRIVE = Path("/content/drive")
else:
    WORK = Path(os.environ.get("LIT_WORK", "/tmp/lit_nb2")); WORK.mkdir(parents=True, exist_ok=True)
    DRIVE = Path(os.environ.get("LIT_FAKE_DRIVE", str(WORK / "drive")))

def sh(cmd, check=True, env=None, cwd=None):
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=cwd, executable="/bin/bash")
    for line in p.stdout:
        print(line, end="", flush=True)
    rc = p.wait()
    if check and rc != 0:
        raise RuntimeError(f"command failed ({rc}): {cmd}")
    return rc

sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo NO GPU", check=False)
''')

code('''
REPO = Path(os.environ.get("LIT_REPO", WORK / "lit-spanglish-asr"))
if not REPO.exists():
    sh(f"git clone --depth 1 {REPO_URL} {REPO}")
else:
    sh("git pull --ff-only || true", check=False, cwd=REPO)
if IN_COLAB and "google.colab" in sys.modules:
    sh('pip install -q "transformers==4.57.6" "peft==0.20.0" "faster-whisper==1.2.1" "ctranslate2==4.8.2" jiwer typer pyyaml soundfile sentencepiece pytest accelerate')
    sh("pip uninstall -y torchao", check=False)   # Colab's torchao 0.10 breaks peft 0.20
    sh("ffmpeg -hide_banner -encoders 2>/dev/null | grep -E 'libopus|libmp3lame' || (apt-get -qq install -y ffmpeg)", check=False)
ENV = dict(os.environ, PYTHONPATH=str(REPO / "src"), TOKENIZERS_PARALLELISM="false")
PY = sys.executable
sh(f"{PY} -m pytest -q -x tests", cwd=REPO, env=ENV)                       # unit tests (CPU + GPU aug on the GPU)
if RUN_E2E_TEST:
    sh(f"LIT_E2E=1 {PY} -m pytest -q -x tests/test_train_e2e.py", cwd=REPO, env=ENV)   # tiny end-to-end training run
''')

code('''
RAW, PREP = WORK / "raw", WORK / "data_prepared"
src = DRIVE / DRIVE_DATA
find = lambda pat: next(iter(sorted(glob.glob(str(src / "**" / pat), recursive=True))), None)
miami_tar, dev_tar = find("*miami*.tar.gz"), find("*enspa_dev*.tar.gz")
assert miami_tar and dev_tar, f"put the two archives under {src}"
cache = DRIVE / DRIVE_RUNS / "prepared_v2_full.tar"
if not (PREP / "train.jsonl").exists():
    if cache.exists():
        sh(f"mkdir -p {PREP} && tar xf {cache} -C {PREP}")
    else:
        RAW.mkdir(exist_ok=True)
        if not (RAW / "miami").exists(): sh(f"tar xzf {miami_tar} -C {RAW}")
if not (RAW / "enspa_dev").exists(): sh(f"tar xzf {dev_tar} -C {RAW}")
if not (PREP / "train.jsonl").exists():
    sh(f"{PY} -m lit.prepare_data --miami_dir {RAW/'miami'} --dev_dir {RAW/'enspa_dev'} --out_dir {PREP} --workers 8", cwd=REPO, env=ENV)
if BUILD_BANK and not (PREP / "train_bank.jsonl").exists():
    sh(f"{PY} scripts/build_codec_bank.py --data_dir {PREP} --variants 3 --workers 10", cwd=REPO, env=ENV)
if not cache.exists():
    cache.parent.mkdir(parents=True, exist_ok=True); sh(f"tar cf {cache} -C {PREP} .")
import collections
for name in ("train", "miami_holdout", "dev"):
    rows = [json.loads(l) for l in open(PREP / f"{name}.jsonl", encoding="utf-8")]
    by = collections.defaultdict(list)
    for r in rows: by[r.get("kind", "dev")].append(r["duration"])
    print(name, {k: f"{len(v)} clips {sum(v)/3600:.2f} h median {sorted(v)[len(v)//2]:.1f}s" for k, v in by.items()})
''')

code('''
# baseline: the OLD model (run1) on the NEW validation sets, so v2 is compared fairly (new convention-aligned refs)
BASE = "openai/whisper-large-v3"
old = DRIVE / DRIVE_RUNS / BASELINE_ADAPTER
if old.exists():
    for split, kind in (("miami_holdout", "turn"), ("miami_holdout", "window"), ("dev", "all")):
        sh(f"{PY} -m lit.evaluate --model {BASE} --adapter {old} --data_dir {PREP} --split {split} --kind {kind} --beams 1 --max_clips 300 2>&1 | grep -E 'wer_post|split|kind'", cwd=REPO, env=ENV, check=False)
else:
    print("no baseline adapter found at", old)
''')

code('''
RUN = DRIVE / DRIVE_RUNS / RUN_NAME
sets = f"data_dir={PREP} out_dir={RUN} dev_in_train={DEV_IN_TRAIN} max_train_minutes={MAX_TRAIN_MINUTES} {EXTRA_SET}"
sh(f"{PY} -u -m lit.train --config {CONFIG} --set {sets}", cwd=REPO, env=ENV)
print((RUN / "final_adapter" / "final.json").read_text()[:900] if (RUN / "final_adapter" / "final.json").exists() else "not finished - re-run to resume")
''')

code('''
# error analysis + rules chosen on dev AND hold-out (a rule must not hurt either)
assert (RUN / "final_adapter").exists(), "training not finished"
for tag, split, kind in (("dev", "dev", "all"), ("holdout", "miami_holdout", "turn")):
    sh(f"{PY} -m lit.evaluate --model {BASE} --adapter {RUN/'final_adapter'} --data_dir {PREP} --split {split} --kind {kind} --beams 1 --max_clips 300 --dump {RUN}/{tag}_preds.csv 2>&1 | grep -E 'split|wer_'", cwd=REPO, env=ENV)
    sh(f"{PY} -m lit.analyze {RUN}/{tag}_preds.csv --top 12 --worst 3 | head -40", cwd=REPO, env=ENV)
sh(f"{PY} scripts/rule_search.py {RUN}/dev_preds.csv {RUN}/holdout_preds.csv --out {RUN}/rules.json", cwd=REPO, env=ENV)
''')

code('''
# export -> submission.zip (built on local disk, then copied) with the chosen decoding config
EXPORT, ZIP = RUN / "export", RUN / "submission.zip"
sh(f"{PY} -m lit.export --model {BASE} --adapter {RUN/'final_adapter'} --data_dir {PREP} --out {EXPORT} --quantization float16", cwd=REPO, env=ENV)
rules = json.loads((RUN / "rules.json").read_text()) if (RUN / "rules.json").exists() else []
(RUN / "infer_final.json").write_text(json.dumps({"language": "es", "beam_size": 5, "compute_type": "float16", "rules": rules}))
sh(f"TMPDIR=/content {PY} scripts/make_submission.py --export {EXPORT} --out {ZIP} --cfg_file {RUN/'infer_final.json'}", cwd=REPO, env=ENV)
''')

code('''
# validate exactly like the platform does (columns, no NaN, logs, timing) + robustness on non-speech / very short clips
sh(f"{PY} scripts/run_submission_local.py --zip {ZIP} --prepared {PREP} --dev_raw {RAW/'enspa_dev'}", cwd=REPO, env=ENV)
sh(f"{PY} scripts/run_submission_local.py --zip {ZIP} --prepared {PREP} --dev_raw {RAW/'enspa_dev'} --n 5 2>&1 | grep -E 'result|VALID|FAIL'", cwd=REPO, env=ENV)
print("zip:", ZIP, f"{ZIP.stat().st_size/1e9:.2f} GB")
''')

code('''
if DOWNLOAD_ZIP and IN_COLAB and "google.colab" in sys.modules:
    from google.colab import files
    files.download(str(ZIP))
''')

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                  "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parents[1] / "notebooks" / "colab_pipeline.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, len(CELLS), "cells")
