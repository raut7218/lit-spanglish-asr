"""Generate notebooks/colab_pipeline.ipynb (Canary pipeline: tests -> data -> convert -> gates G0/G1 -> train -> G2 +
analysis -> zip -> validate)."""

import json
from pathlib import Path

CELLS = []


def md(s):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md("""
# Lost in Transcription — Canary-1b-v2 pipeline (Colab A100)

Tests → data → **convert** canary-1b-v2 to our NeMo-free port → **G0** parity with NeMo → **G1** zero-shot dev WER →
**full fine-tune** (frozen lower encoder, verbatim style token, per-clip language token, GPU aug, EMA) →
**G2** dev WER must beat the old Whisper system (0.0875) → rules chosen on dev **and** hold-out → `submission.zip` →
validation. Everything is resumable from Google Drive.

**Final model:** once the recipe is chosen, re-run training with `DEV_IN_TRAIN = "all"`, a new `RUN_NAME` and the
step count that won (`EXTRA_SET = "max_steps=..."`); dev WER is then contaminated, so the hold-out is the only check.
""")

code('''
RUN_NAME    = "canary1"
REPO_URL    = "https://github.com/raut7218/lit-spanglish-asr.git"
DRIVE_DATA  = "MyDrive/lit_data"
DRIVE_RUNS  = "MyDrive/lit_runs"
CONFIG      = "configs/canary_a100.yaml"
EXTRA_SET   = ""             # e.g. "max_steps=3000 freeze_encoder_below=8 verbatim=false"
DEV_IN_TRAIN = "none"        # none | spk1 | spk2 | all   (all = final fit; dev WER is then contaminated)
BUILD_BANK  = False          # optional offline codec bank: SLOW (~1.5-2 h). Training already applies an online Opus->MP3 chain
RUN_G0      = True           # NeMo parity (installs NeMo in a separate venv once, ~10 min; cached on Drive)
RUN_G1      = True           # zero-shot dev WER of the converted base model
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
    WORK = Path(os.environ.get("LIT_WORK", "/tmp/lit_nb")); WORK.mkdir(parents=True, exist_ok=True)
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
    # the competition runtime pins transformers 4.57.x (ParakeetEncoder) and has no NeMo: train with the same stack
    sh('pip install -q "transformers==4.57.6" sentencepiece safetensors jiwer pyyaml soundfile librosa pytest')
    sh("ffmpeg -hide_banner -encoders 2>/dev/null | grep -E 'libopus|libmp3lame' || (apt-get -qq install -y ffmpeg)", check=False)
ENV = dict(os.environ, PYTHONPATH=str(REPO / "src"), TOKENIZERS_PARALLELISM="false")
PY = sys.executable
sh(f"{PY} -m pytest -q -x tests", cwd=REPO, env=ENV)   # unit tests incl. a tiny end-to-end train -> export -> runtime run
''')

code('''
RAW, PREP = WORK / "raw", WORK / "data_prepared"
src = DRIVE / DRIVE_DATA
find = lambda pat: next(iter(sorted(glob.glob(str(src / "**" / pat), recursive=True))), None)
miami_tar, dev_tar = find("*miami*.tar.gz"), find("*enspa_dev*.tar.gz")
assert miami_tar and dev_tar, f"put the two archives under {src}"
cache = DRIVE / DRIVE_RUNS / "prepared_v2_full.tar"
RAW.mkdir(exist_ok=True)   # raw dev audio is needed even when prepared data comes from the cache
if not (PREP / "train.jsonl").exists():
    if cache.exists():
        sh(f"mkdir -p {PREP} && tar xf {cache} -C {PREP}")
    else:
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
# convert canary-1b-v2 (.nemo from the HF hub) -> NeMo-free model dir; cached on Drive, used from local disk
BASE = WORK / "canary_base"
BASE_DRIVE = DRIVE / DRIVE_RUNS / "canary_base"
if not (BASE / "model.safetensors").exists():
    if (BASE_DRIVE / "model.safetensors").exists():
        shutil.copytree(BASE_DRIVE, BASE, dirs_exist_ok=True)
    else:
        sh(f"{PY} scripts/convert_canary.py --out {BASE}", cwd=REPO, env=ENV)
        shutil.copytree(BASE, BASE_DRIVE, dirs_exist_ok=True)
print(json.loads((BASE / "config.json").read_text())["verbatim_token"], "=", "verbatim style token")
''')

code('''
# G0: parity with NeMo on 6 dev clips. NeMo lives in its own venv so it cannot touch the pinned training stack.
if RUN_G0:
    if not (BASE_DRIVE / "golden.pt").exists():
        dev_rows = [json.loads(l) for l in open(PREP / "dev.jsonl", encoding="utf-8")][:6]
        clips = " ".join(str(PREP / r["audio"]) for r in dev_rows)
        NEMO_PY = "/content/nemo_env/bin/python"
        if not os.path.exists(NEMO_PY):
            sh("pip install -q uv && uv venv -q /content/nemo_env && uv pip install -q --python /content/nemo_env/bin/python 'nemo_toolkit[asr]' soundfile")
        sh(f"PYTHONPATH={REPO/'src'} {NEMO_PY} scripts/convert_canary.py --out {BASE} --golden {clips}", cwd=REPO)
        shutil.copy(BASE / "golden.pt", BASE_DRIVE / "golden.pt")
    elif not (BASE / "golden.pt").exists():
        shutil.copy(BASE_DRIVE / "golden.pt", BASE / "golden.pt")
    sh(f"LIT_CANARY_DIR={BASE} {PY} -m pytest -q -s tests/test_canary_parity.py", cwd=REPO, env=ENV)
''')

code('''
# G1: zero-shot dev WER of the base model (stock prompt), per language prompt and with the dual-prompt pick.
# Reference: Whisper large-v3 zero-shot was 0.51 (es) / 0.43 (en).
if RUN_G1:
    for langs in ("es", "en", "es en"):
        sh(f"{PY} -m lit.evaluate --model {BASE} --data_dir {PREP} --split dev --languages {langs} --no_verbatim 2>&1 | grep -E 'languages|wer_'", cwd=REPO, env=ENV, check=False)
''')

code('''
RUN = DRIVE / DRIVE_RUNS / RUN_NAME
sets = f"model={BASE} data_dir={PREP} out_dir={RUN} dev_in_train={DEV_IN_TRAIN} max_train_minutes={MAX_TRAIN_MINUTES} {EXTRA_SET}"
sh(f"{PY} -u -m lit.train --config {CONFIG} --set {sets}", cwd=REPO, env=ENV)
print((RUN / "final_model" / "final.json").read_text()[:900] if (RUN / "final_model" / "final.json").exists() else "not finished - re-run to resume")
''')

code('''
# G2 + error analysis + rules chosen on dev AND hold-out (a rule must not hurt either). G2: dev must beat 0.0875.
FINAL = RUN / "final_model"
assert (FINAL / "model.safetensors").exists(), "training not finished"
for tag, split, kind in (("dev", "dev", "all"), ("holdout", "miami_holdout", "turn")):
    sh(f"{PY} -m lit.evaluate --model {FINAL} --data_dir {PREP} --split {split} --kind {kind} --beams 4 --max_clips 300 --dump {RUN}/{tag}_preds.csv 2>&1 | grep -E 'split|wer_'", cwd=REPO, env=ENV)
    sh(f"{PY} -m lit.analyze {RUN}/{tag}_preds.csv --top 12 --worst 3 | head -40", cwd=REPO, env=ENV)
sh(f"{PY} scripts/rule_search.py {RUN}/dev_preds.csv {RUN}/holdout_preds.csv --out {RUN}/rules.json", cwd=REPO, env=ENV)
''')

code('''
# submission.zip (built on local disk, then copied) with the chosen decoding config
ZIP = RUN / "submission.zip"
rules = json.loads((RUN / "rules.json").read_text()) if (RUN / "rules.json").exists() else []
(RUN / "infer_final.json").write_text(json.dumps({"languages": ["es", "en"], "beam_size": 4, "rules": rules}))
sh(f"TMPDIR=/content {PY} scripts/make_submission.py --export {FINAL} --out {ZIP} --cfg_file {RUN/'infer_final.json'}", cwd=REPO, env=ENV)
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
