# lit-spanglish-asr

Spanish–English code-switching ASR for the **Lost in Transcription** competition (Mozilla Data Collective / DrivenData).
Metric: WER after the organisers' normaliser (`score.py`). Train on Google Colab, run offline in the competition runtime.

## Architecture (and why it differs from the planning doc)

| Decision | Choice | Evidence |
|---|---|---|
| Base model | **Whisper large-v3-turbo** (T4) / **large-v3** (A100) | Runtime pins `transformers 4.57.6`, has **no NeMo** → Canary-1b-v2 cannot load there. Runtime *does* ship `faster-whisper 1.2.1` + `ctranslate2 4.8.2` (`runtime/uv.lock`). |
| Fine-tuning | LoRA r=32 on all attention + MLP projections (encoder **and** decoder), fp16/bf16 base frozen, gradient checkpointing | fits a 16 GB T4; encoder LoRA adapts to phone-mic/codec audio |
| Targets | Miami CHAT → verbatim words, **scorer-normalised** (`lit.normalize` = verbatim copy of `score.py`) | scorer lowercases only sentence-initial letters, so casing is learned from data |
| Prompt | fixed `<|es|><|transcribe|><|notimestamps|>` | test is mixed; `en` is compared on dev (`LANGUAGE` in the notebook) |
| Augmentation | Opus/MP3/AMR-NB round-trips (ffmpeg), coloured + babble noise, synthetic reverb, band-limit, speed, clipping, SpecAugment | Miami = belt mics, test = WhatsApp voice notes |
| Validation | 35-min official dev set (honest) + speaker-disjoint Miami hold-out | dev never trained on until the optional last step |
| Inference | merge LoRA → CTranslate2 → `faster-whisper` `BatchedInferencePipeline`; ≤29.5 s clips decoded whole, longer ones VAD-chunked; loop/hallucination guard; data-driven casing fix | offline, no torch needed at inference |

Column name: the platform CSV column is **`transcript`** (not `transcription`).

## Layout
```
src/lit/  normalize.py chat.py prepare_data.py augment.py features.py model_utils.py train.py evaluate.py
          export.py infer.py casing.py postprocess.py
submission_src/main.py         # runtime entrypoint (copied to zip root by make_submission.py)
scripts/  make_submission.py run_submission_local.py smoke_local.sh build_notebook.py official_score.py
configs/  smoke.yaml colab_t4.yaml colab_a100.yaml
notebooks/colab_pipeline.ipynb # the end-to-end pipeline for Colab
tests/
```

## Run on Colab
1. Upload `*miami.tar.gz` and `*enspa_dev.tar.gz` to `Drive/lit_data/`.
2. Open `notebooks/colab_pipeline.ipynb` in Colab (GPU runtime), set `REPO_URL`, run with `SMOKE=True`, then `SMOKE=False`.
3. Upload the resulting `submission.zip` on the competition page (do a platform smoke test first).

## Local smoke test (tiny model, any machine with ffmpeg)
```
pip install -e '.[dev]'
scripts/smoke_local.sh /path/to/miami /path/to/enspa_dev
```

## Rules respected
Only organiser-provided data for training; no competition audio/text is sent to any hosted API; data and weights are git-ignored.

## Results so far (dev = the 35-min official set, never trained on; scorer-exact WER)

| System | Dev WER |
|---|---|
| Whisper large-v3 zero-shot, `es` token / `en` token, greedy | 0.509 / 0.430 |
| + LoRA fine-tune on Bangor Miami (500 steps, greedy) | 0.1051 |
| + convention rules (`gonna`->`going to`, capital `I`, `ah`->`uh`, drop `um`), greedy | 0.0919 |
| **Shipped zip: CTranslate2 fp16, beam 5, rules, no casing lexicon (155 clips, 0.52 s/clip on A100)** | **0.0875** |

Caveats: dev is only 155 clips (~+-0.01); the 4 rules and the checkpoint were chosen while looking at dev. Speaker-disjoint Miami hold-out WER was 0.128-0.143 (different transcription conventions).
Error analysis (`python -m lit.analyze`) showed the remaining gap is mostly convention mismatch: Miami writes "gonna" (820x) / "ah" (603x), dev writes "going to" / never "ah".

## Speed notes (A100 40 GB)
Training loop: gradient checkpointing off with automatic OOM fallback, 10 dataloader workers, TF32, fused AdamW -> 3.6 s/step (was 6.2 s). GPU utilisation is ~45-70%: the remaining bottleneck is Python/launch overhead in the LoRA-wrapped model, not data.

## Next ideas
Fix conventions in the training targets (gonna->going to, ah) and retrain; finish the cosine schedule (steps 500-1107); language token `en` check on the fine-tuned model; final retrain including dev only after model selection.
