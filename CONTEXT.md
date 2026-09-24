# CONTEXT — Lost in Transcription (Spanish–English code-switching ASR)

Single source of truth for this repo: the competition, every experiment so far with its score, where we stand, and
what the live pipeline (**Candidate A**) is. Everything not needed for Candidate A was deleted on 2026-09-24
(old branches/tag are in `../lit-spanglish-asr-archive.bundle` on the author's Desktop: `git clone <bundle>` to
recover).

## 1. Competition

| | |
|---|---|
| Name | Lost in Transcription (Mozilla Data Collective / DrivenData) |
| Task | Transcribe Spanish–English code-switched speech (WhatsApp-style voice notes), verbatim, cased |
| Metric | WER after the organisers' normaliser (`scripts/official_score.py`, copied exactly in `src/lit/normalize.py`). Punctuation removed; casing **is** scored except the sentence-initial letter |
| Test set | 592 clips, 1 s – ~4 min |
| Training data given | Bangor Miami corpus (~25 h, CHAT transcripts, belt mics, median SNR ~6 dB) + official dev set (155 clips, 35 min, test-like voice notes, median SNR ~22 dB) |
| Rules | External data and open-weight models allowed (external data must be publishable to MDC). **No hosted APIs** on competition data. Runtime is offline |
| Runtime | Code submission: zip with `main.py` at root. A100 80 GB, 24 vCPU, 220 GB RAM, **2 h limit**, no network. Pinned: torch 2.11.0+cu130, transformers 4.57.6, peft 0.20.0, faster-whisper 1.2.1, ctranslate2 4.8.2, vllm 0.23.0, qwen-asr 0.0.6, kenlm, pyctcdecode |
| Logs | ≤500 lines × 300 chars, must not contain clip names / test info |
| Output | `submission.csv`, columns exactly `audio_filename,transcript`, no empty / NaN transcripts |
| Submissions | 3 per rolling 7 days. Next slot **2026-09-27 UTC**. Deadline **2026-10-02** |
| Zip size | Undocumented. 6.3 GB accepted; Candidate A is 8.0 GB (untested on the platform) |

Leaderboard on 2026-09-24: #1 0.1013, #2 0.1033, #3 0.1034 … **we are #15 at 0.1178** (team `jnvgkraut`). Goal: < 0.10.

## 2. Timeline of experiments and scores

Offline sets (all through the exact runtime decoding path):
**dev** = official dev, 155 clips (±~0.01 noise); **hold-out** = speaker-disjoint Miami turns (first 300);
**long** = dev clips concatenated into 60–240 s clips (tests the long-form path); **macro** = mean of the three.

| Date | Pipeline | dev | hold-out | long | Leaderboard |
|---|---|---|---|---|---|
| 09-19/20 | Whisper large-v3 zero-shot (`es` / `en` token) | 0.509 / 0.430 | | | |
| 09-20 | v1 Whisper large-v3 LoRA (Miami), greedy | 0.1051 | | | |
| 09-20 | v1 + convention rules, beam 5, CT2 fp16 (`run2` lineage) | 0.0875 | ~0.18 | 0.095 | **0.1181** (#11) |
| 09-21 | v3-lora-scale-stage2: checkpoint + rules tuned on dev | ~0.081 | | | **~0.118** |
| 09-23 | Canary-1b-v2 full fine-tune (NeMo-free port) | 0.138 greedy | | | abandoned (translates code-switched speech, overfits) |
| 09-23 | v3 fixes: CHAT matrix-language bug, zero-shot label filter, EMA warm-up; lv3 step 100 | 0.0919 | 0.1644 | 0.0961 | |
| 09-23 | **5 Whisper models, MBR medoid + rules** (`v3_mbr5`) | 0.0841 | 0.1589 | 0.0925 | **0.1178** (#15 on 09-24) — current best |
| 09-24 | Qwen3-ASR-1.7B zero-shot, language auto | 0.243 | 0.250 | 0.213 | |
| 09-24 | qwen1: LoRA lr 1e-4, avg steps 100+200 | 0.1095 | 0.1938 | 0.1149 | |
| 09-24 | qwen2: LoRA lr 5e-5, avg steps 100/150/200 | 0.1035 | 0.1861 | 0.1135 | |
| 09-24 | **Candidate A**: qwen2 + 3 Whisper, ROVER | 0.0839 | **0.1520** | **0.0891** | not submitted yet |

Individual Whisper systems (beam 5, rules, int8_float16), as numbered in the `v3_mbr5` zip:

| zip name (v3_mbr5) | Run on Drive | Recipe | dev | hold-out | long | In Candidate A as |
|---|---|---|---|---|---|---|
| ct2 | `lv3` step 100 | v3, large-v3 | 0.0873 | 0.1678 | 0.0933 | – |
| ct2_2 | `lv3` checkpoint average | v3, large-v3 | 0.0859 | 0.1614 | 0.0945 | `ct2` |
| ct2_3 | `turbo2` | v3, large-v3-turbo | 0.0941 | 0.1695 | 0.1003 | `ct2_2` |
| ct2_4 | `turbo1` | v3, turbo, before data cleaning | 0.0943 | 0.1738 | 0.0985 | – |
| ct2_5 | `run2` | v1/v2, large-v3 (`configs/colab_a100_v2.yaml`) | 0.0867 | 0.1870 | 0.0897 | `ct2_3` |

Ensemble search (09-24, 7 systems, every subset, MBR vs ROVER): best 12 ROVER combos span macro 0.1081–0.1088
(a tie, far inside the noise). All-7 ROVER 0.1082; shipped 5-Whisper MBR 0.1118; Candidate A 0.1084.

## 3. Lessons (what the logs say)

1. **Offline gains have not reached the leaderboard.** From v1 to v3_mbr5: dev −4 to −8 %, hold-out −12 %,
   leaderboard 0.1181 → 0.1178. The leaderboard sits ~0.034 above dev every time. The test set differs from dev and
   Miami in a way we do not measure (candidates: long clips, recording conditions, transcription conventions).
2. **155 dev clips = ±0.01.** Do not select on dev alone; never tune on dev (stage2 lesson). Judge on dev + hold-out
   + long together, and prefer changes big enough to be visible.
3. **Every model plateaus after ~100 training steps** on Miami (Whisper and Qwen); longer training fits Miami's
   conventions and hurts dev. Data, not steps, is the ceiling.
4. **Diversity beats more of the same.** MBR over one model's es/en/auto decodes did nothing; different models help;
   a different *family* (Qwen) is in every top combination although it is the weakest single system.
5. **ROVER (word-level vote) > MBR medoid (whole-transcript pick)**, mostly on long clips.
6. **Conventions matter as much as acoustics**: rules (`capital_i`, `ok_to_okay`, `ah_to_uh`) gave dev −6 % but
   hold-out +2 % (Miami writes `gonna`, `ah`; dev writes `going to`, never `ah`).
7. Zero-shot multilingual models with a single target language (Canary, Qwen) **translate** code-switched speech.
8. Fine-tuned Qwen stops early on long audio: `lit.qwen` cuts clips > 29 s at the quietest point (long 0.18 → 0.115).

## 4. Candidate A (the live pipeline)

```
clip ─┬─ Qwen3-ASR-1.7B + LoRA (qwen2, merged)  vLLM, own process, >29 s cut at silence ─┐
      ├─ Whisper large-v3 LoRA (lv3 avg)         faster-whisper CT2 int8_float16, beam 5 ├─ rules ─ ROVER ─ finalize ─ CSV
      ├─ Whisper large-v3-turbo LoRA (turbo2)    "                                       │  (pivot = MBR medoid,
      └─ Whisper large-v3 LoRA (run2)            "                                       ┘   word vote, ties → pivot)
```

- Config: `configs/infer_cand_A.json` (`systems: [qwen, ct2, ct2_2, ct2_3]`, `combine: rover`).
- Zip: Drive `lit_runs/cand_A_qwen2_ct2x3_rover/submission.zip` (8.0 GB, 38 files) + `ens_table.csv`.
- Local runtime check passed: 155 dev clips, dev 0.0843, 480 s (→ ~30 min for 592 clips), 19 log lines.
- Safety: if Qwen fails at runtime, `main.py` logs it and continues with the Whisper systems; a time guard
  (`ensemble_budget_s` 4800) stops adding systems that would not fit in 2 h.

Rebuild the zip (Colab, Drive mounted; the Whisper exports are the `model/ct2_*` folders of the old v3_mbr5 zip or
`python -m lit.export` of the Drive runs):
```
python scripts/make_submission.py --export EXP_lv3avg --extra EXP_turbo2 EXP_run2 \
    --qwen /content/drive/MyDrive/lit_runs/qwen2/merged --cfg_file configs/infer_cand_A.json --out cand_A.zip
python scripts/run_submission_local.py --zip cand_A.zip --prepared /content/data_v3 --dev_raw RAW/enspa_dev
```

## 5. Code map

| Path | Role |
|---|---|
| `submission_src/main.py` | Runtime entry: runs each system, combines (ROVER/MBR), writes + self-checks the CSV |
| `src/lit/qwen.py` | Qwen3-ASR runtime/eval (vLLM, chunking, target format `language X<asr_text>text`) |
| `src/lit/train_qwen.py` | Qwen LoRA trainer (left-padding label mask, GPU aug, EMA, `--finalize`, merge for vLLM) — `configs/qwen_a100.yaml` |
| `src/lit/train.py`, `export.py` | Whisper LoRA trainer → merge → CTranslate2 — `configs/turbo_a100.yaml` (v3), `configs/colab_a100_v2.yaml` (run2) |
| `src/lit/infer.py` | faster-whisper runtime (beam, VAD-chunked long form, loop guard) |
| `src/lit/mbr.py` | `mbr_pick` (medoid), `align`, `rover`; `python -m lit.mbr a.csv b.csv …` scores combinations |
| `src/lit/prepare_data.py`, `chat.py`, `conventions*`, `quality.py` | Miami CHAT → clips/windows with dev conventions; zero-shot label filter |
| `src/lit/augment.py`, `gpu_aug.py`, `features.py`, `audio.py` | Codec chain, noise/reverb/EQ on GPU, log-mel, long-clip splitting |
| `src/lit/postprocess.py`, `rules.py`, `casing.py`, `normalize.py` | Scorer-exact normaliser, convention rules, final CSV text |
| `src/lit/analyze.py`, `scripts/rule_search.py` | Error analysis and rule selection on prediction dumps |
| `scripts/decode_eval.py`, `qwen_eval.py` | Score Whisper / Qwen on dev, hold-out, long; dump id/ref/hyp CSVs |
| `scripts/make_submission.py`, `run_submission_local.py` | Build the zip; validate it end to end like the platform |
| `notebooks/colab_pipeline.ipynb` (from `scripts/build_notebook.py`) | Whisper v3 train → export → eval → zip on Colab |

Tests: `PYTHONUTF8=1 pytest -q` (Qwen tests skip without `qwen_asr`).

## 6. Google Drive (`MyDrive/`)

| Path | What |
|---|---|
| `lit_data/` | Raw competition archives (Miami, enspa_dev) |
| `lit_runs/prepared_v3.tar`, `quality_v3.jsonl` | Prepared data cache + label-check cache used by every current trainer |
| `lit_runs/lv3/`, `turbo2/`, `run2/` | Whisper runs behind Candidate A |
| `lit_runs/qwen2/` | Qwen run behind Candidate A (`merged/` is what ships) |
| `lit_runs/dumps/` | Per-system predictions on dev / hold-out / long (id, ref, hyp) for Candidate A's systems |
| `lit_runs/cand_A_qwen2_ct2x3_rover/` | The Candidate A zip + ensemble table |

## 7. Colab setup (A100 40 GB, via colab-mcp)

- colab-mcp drives its **own** notebook tab: that tab's runtime must be set to A100 (or CPU for analysis).
- vLLM 0.23 is a CUDA-13 build: `pip install --force-reinstall torch==2.11.0+cu130 torchaudio==2.11.0+cu130
  torchvision==0.26.0+cu130` (`--index-url https://download.pytorch.org/whl/cu130`); prepend
  `/usr/local/lib/python3*/dist-packages/nvidia/cu13/lib` (glob it) to `LD_LIBRARY_PATH`;
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. GPU aug on 40 GB needs batch 6 × grad-accum 8.
- Cells running > ~100 s lose their output: launch jobs in the background with a log file and poll with short cells.
- After `runtime.unassign()`, running ANY cell connects a new GPU (and spends units).
- Budget: Colab Pro, ~100 compute units left on 2026-09-24 (A100 ≈ 5.3 units/h). CPU work (analysis) on a CPU runtime.

## 8. Next steps (ranked)

1. **2026-09-27: submit Candidate A.** It is the first submission that differs in kind (second model family + ROVER).
   ≤ 0.114 → diversity transfers: add more *different* systems. ≈ 0.118 → the gap is domain/conventions/long audio:
   stop ensembling.
2. **CPU only, free**: weighted ROVER (`lit.mbr.rover(weights=…)`), bootstrap confidence intervals, oracle WER of the
   candidate pool (gates the n-best + selector idea), error breakdown on hold-out (`lit.analyze`).
3. Final retrain with the 155 dev clips included (only test-like labelled data; training on it ≠ tuning on it).
4. Qwen trained on longer windows (test clips up to 4 min), ~2 A100-h.
5. n-best + **audio-conditioned** rescoring (Qwen/Whisper log-prob of each candidate, few weights tuned on hold-out),
   only if the oracle shows headroom. A text-only LLM selector is not recommended (prefers monolingual text, no clean
   training data).
