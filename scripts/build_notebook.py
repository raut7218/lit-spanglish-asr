"""Generate notebooks/colab_pipeline.ipynb (v3: turbo, cleaned data, beam-5 runtime).

Cells: params -> setup (Drive, repo branch, deps, data cache) -> quality audit -> train -> export + decoding check
(dev / Miami hold-out / synthetic long clips through the shipped faster-whisper path) -> submission.zip -> validation.
Every step is cached/resumable on Drive; long jobs log to /content/logs.
"""

import json
from pathlib import Path

CELLS = []


def md(s):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md("""
# Lost in Transcription (es-en) — v3 pipeline, Colab A100
Whisper large-v3-turbo + LoRA, Miami data with the matrix-language fix and a zero-shot label filter, EMA with
warm-up, beam-5 faster-whisper runtime. Resumable from Google Drive.
""")

code('''
RUN_NAME    = "turbo1"
BRANCH      = "main"
REPO_URL    = "https://github.com/raut7218/lit-spanglish-asr.git"
CONFIG      = "configs/turbo_a100.yaml"
EXTRA_SET   = ""             # e.g. "max_steps=600 dev_in_train=all quality.max_zs_wer=0"
RUN_QUALITY = True           # zero-shot label check (~30-60 min once; cached on Drive)
INFER_CFG   = {"language": "es", "beam_size": 5, "compute_type": "float16", "rules": ["capital_i"]}
''')

code('''
import os, sys, json, glob, subprocess, time
from pathlib import Path
from google.colab import drive
drive.mount("/content/drive")
DRIVE = Path("/content/drive/MyDrive")
REPO, RAW, DATA = Path("/content/lit-spanglish-asr"), Path("/content/raw"), Path("/content/data_v3")
ENV = dict(os.environ, PYTHONPATH=str(REPO / "src"), TOKENIZERS_PARALLELISM="false")
PY = sys.executable

def sh(cmd, check=True, cwd=None):
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd or REPO, env=ENV, executable="/bin/bash")
    for line in p.stdout: print(line, end="", flush=True)
    if p.wait() and check: raise RuntimeError(f"failed: {cmd}")

if not REPO.exists(): sh(f"git clone -b {BRANCH} {REPO_URL} {REPO}", cwd="/content")
else: sh("git pull -q")
sh('pip install -q "transformers==4.57.6" "peft==0.20.0" "faster-whisper==1.2.1" "ctranslate2==4.8.2" jiwer pyyaml soundfile accelerate 2>&1 | grep -v "dependency conflicts" | tail -2')
sh("pip uninstall -y -q torchao", check=False)
src = DRIVE / "lit_data"
miami_tar = sorted(glob.glob(str(src / "**" / "*miami*.tar.gz"), recursive=True))[0]
dev_tar = sorted(glob.glob(str(src / "**" / "*enspa_dev*.tar.gz"), recursive=True))[0]
cache = DRIVE / "lit_runs" / "prepared_v3.tar"
RAW.mkdir(exist_ok=True)
if not (RAW / "enspa_dev").exists(): sh(f"tar xzf {dev_tar} -C {RAW}")
if not (DATA / "train.jsonl").exists():
    if cache.exists():
        sh(f"mkdir -p {DATA} && tar xf {cache} -C {DATA}")
    else:
        if not (RAW / "miami").exists(): sh(f"tar xzf {miami_tar} -C {RAW}")
        sh(f"{PY} -m lit.prepare_data --miami_dir {RAW/'miami'} --dev_dir {RAW/'enspa_dev'} --out_dir {DATA} --workers 10")
        sh(f"tar cf {cache} -C {DATA} .")
sh(f"wc -l {DATA}/*.jsonl")
''')

code('''
# quality audit: VAD speech ratio / SNR + best-of(es,en) zero-shot WER per clip -> DATA/quality.jsonl (train filter)
qcache = DRIVE / "lit_runs" / "quality_v3.jsonl"
if qcache.exists() and not (DATA / "quality.jsonl").exists(): sh(f"cp {qcache} {DATA}/quality.jsonl")
if RUN_QUALITY and not qcache.exists():
    sh(f"{PY} -m lit.quality --data_dir {DATA} --language es,en --batch_size 96 --workers 10 2>&1 | grep -vi warn")
    sh(f"cp {DATA}/quality.jsonl {qcache}")
''')

code('''
RUN = DRIVE / "lit_runs" / RUN_NAME
sh(f"{PY} -u -m lit.train --config {CONFIG} --set data_dir={DATA} out_dir={RUN} {EXTRA_SET} 2>&1 | grep -vi warn")
print((RUN / "final_adapter" / "final.json").read_text()[:800])
''')

code('''
# export (merge LoRA -> CTranslate2 fp16) and score the SHIPPED decoding path
cfgj = json.loads((RUN / "config.json").read_text())
EXPORT = RUN / "export"
if not (EXPORT / "ct2" / "model.bin").exists():
    sh(f"{PY} -m lit.export --model {cfgj['model']} --adapter {RUN/'final_adapter'} --data_dir {DATA} --out {EXPORT}")
grid = [{"beam_size": 1, "rules": []}, {"beam_size": 5, "rules": []}, dict(INFER_CFG)]
sh(f"""{PY} scripts/decode_eval.py --ct2 {EXPORT/'ct2'} --data_dir {DATA} --sets dev miami_holdout long --grid '{json.dumps(grid)}' 2>&1 | grep -E "decode\\\\]|Error" """)
''')

code('''
ZIP = RUN / "submission.zip"
(RUN / "infer_final.json").write_text(json.dumps(INFER_CFG))
sh(f"TMPDIR=/content {PY} scripts/make_submission.py --export {EXPORT} --out {ZIP} --cfg_file {RUN/'infer_final.json'}")
sh(f"{PY} scripts/run_submission_local.py --zip {ZIP} --prepared {DATA} --dev_raw {RAW/'enspa_dev'}")
''')

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                  "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parents[1] / "notebooks" / "colab_pipeline.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, len(CELLS), "cells")
