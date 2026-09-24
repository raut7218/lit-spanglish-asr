"""LoRA fine-tuning of Qwen3-ASR (1.7B) on the same prepared data as lit.train (Whisper).

    python -m lit.train_qwen --config configs/qwen_a100.yaml [--set key=value ...]

Reused from lit.train: quality filter, language-matched sampler, cosine LR, EMA (with warm-up), macro(hold-out, dev)
selection. Augmentation runs per clip in the dataloader workers (speed, Opus->MP3 codec chain, lit.gpu_aug on CPU)
because Qwen's processor computes the features on the CPU. Targets: lit.qwen.target_text ("language X<asr_text>...").
The best (EMA) adapter is merged and saved as a full HF model dir that qwen-asr / vLLM load directly (`merged/`).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset

from .audio import load_audio
from .augment import codec_chain
from .gpu_aug import GpuAugConfig, GpuAugmenter
from .model_utils import read_manifest
from .normalize import wer
from .postprocess import postprocess
from .qwen import target_text
from .train import EMA, adapter_state, average_adapters, cosine_lr, quality_filter, sampling_weights, selection_score

DEFAULTS = dict(
    model="Qwen/Qwen3-ASR-1.7B",
    data_dir="/content/data_v3",
    out_dir="/content/drive/MyDrive/lit_runs/qwen1",
    lora=dict(r=64, alpha=64, dropout=0.05, audio_from_layer=12, text=True),
    lr=1e-4, min_lr=1e-6, warmup_steps=30, weight_decay=1e-3, grad_clip=1.0,
    batch_size=16, grad_accum=3, max_steps=600, epoch_samples=14400,
    eval_steps=100, keep_best=2, eval_batch_size=32, max_train_minutes=0,
    val=dict(holdout_turn=200, holdout_window=60),
    sampler=dict(kind_share=dict(turn=0.35, window=0.60, nonspeech=0.015), dev_share=0.08, match_dev_language=True),
    dev_in_train="none",
    ema_decay=0.999,
    num_workers="auto", gradient_checkpointing=False,
    cpu_aug=dict(p_speed=0.4, speeds=[0.9, 1.0, 1.1], p_codec=0.3),
    gpu_aug=dict(), seed=13, max_clip_seconds=29.5,
    quality=dict(max_zs_wer=0.8, min_snr_db=-99.0, max_len_ratio=0.0),
)
HF_FILES = ["config.json", "generation_config.json", "preprocessor_config.json", "processor_config.json",
            "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json", "chat_template.json", "merges.txt",
            "vocab.json"]


def load_config(path, overrides=()):
    cfg = json.loads(json.dumps(DEFAULTS))
    if path:
        for k, v in (yaml.safe_load(open(path)) or {}).items():
            cfg[k] = {**cfg[k], **v} if isinstance(cfg.get(k), dict) and isinstance(v, dict) else v
    for kv in overrides:
        k, v = kv.split("=", 1)
        d, keys = cfg, k.split(".")
        for kk in keys[:-1]:
            d = d[kk]
        d[keys[-1]] = yaml.safe_load(v)
    return cfg


def lora_targets(cfg, n_audio: int) -> str:
    lc = cfg["lora"]
    parts = []
    aud = "|".join(str(i) for i in range(lc["audio_from_layer"], n_audio))
    if aud:
        parts.append(rf"thinker\.audio_tower\.layers\.({aud})\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)|fc1|fc2)")
    if lc["text"]:
        parts.append(r"thinker\.model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))")
    return "|".join(parts)


class QwenClips(Dataset):
    def __init__(self, rows, root, cfg):
        self.rows, self.root, self.cfg = rows, Path(root), cfg
        self.aug = GpuAugmenter(GpuAugConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg["gpu_aug"].items()}))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r, ca = self.rows[i], self.cfg["cpu_aug"]
        rng = np.random.default_rng((self.cfg["seed"], i, int(time.time() * 1e3) % 100000))
        audio = load_audio(self.root / r["audio"])
        if rng.random() < ca["p_speed"]:
            f = float(rng.choice(ca["speeds"]))
            if abs(f - 1.0) > 1e-3:
                n = int(round(len(audio) / f))
                audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
        audio = audio[: int(self.cfg["max_clip_seconds"] * 16000)]
        if not r.get("nonspeech"):
            if rng.random() < ca["p_codec"]:
                audio = codec_chain(audio, rng)
            with torch.no_grad():
                audio = self.aug(torch.from_numpy(audio)[None], torch.tensor([len(audio)]))[0].numpy()
        else:  # real silence/noise clips stay quiet (level ~ -30..-50 dB)
            audio = audio * (10 ** (-rng.uniform(30, 50) / 20) / (np.sqrt(np.mean(audio**2)) + 1e-8))
        return audio.astype(np.float32), target_text(r)


class Collate:
    """Qwen SFT collation (prefix = chat template up to the assistant turn; loss only on the target + eos)."""

    def __init__(self, processor):
        self.p = processor
        msgs = [{"role": "system", "content": ""}, {"role": "user", "content": [{"type": "audio", "audio": None}]}]
        self.prefix = processor.apply_chat_template([msgs], add_generation_prompt=True, tokenize=False)[0]
        self.eos = processor.tokenizer.eos_token or ""

    def __call__(self, batch):
        audios, targets = zip(*batch)
        full = self.p(text=[self.prefix + t + self.eos for t in targets], audio=list(audios), return_tensors="pt", padding=True)
        pre = self.p(text=[self.prefix] * len(audios), audio=list(audios), return_tensors="pt", padding=True)
        labels = full["input_ids"].clone()
        for i, n in enumerate(pre["attention_mask"].sum(1).tolist()):
            labels[i, :n] = -100
        labels[full["attention_mask"] == 0] = -100
        full["labels"] = labels
        return full


def run_eval(asr, sets, root, bs):
    out = {}
    for name, rows in sets.items():
        if not rows:
            continue
        hyps = []
        for i in range(0, len(rows), bs):
            chunk = [(load_audio(root / r["audio"]), 16000) for r in rows[i : i + bs]]
            hyps += [postprocess(x.text) for x in asr.transcribe(audio=chunk)]
        out[name] = wer([r.get("ref", r["text"]) for r in rows], hyps)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--dry", action="store_true", help="build everything, run 1 step + eval + merge, exit")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    dtype = torch.bfloat16 if cuda else torch.float32

    from qwen_asr import Qwen3ASRModel

    asr = Qwen3ASRModel.from_pretrained(cfg["model"], dtype=dtype, device_map=str(device),
                                        max_inference_batch_size=cfg["eval_batch_size"], max_new_tokens=512)
    model, processor = asr.model, asr.processor
    n_audio = model.config.thinker_config.audio_config.encoder_layers
    lc = cfg["lora"]
    pm = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                          target_modules=lora_targets(cfg, n_audio), bias="none"))
    pm.print_trainable_parameters()  # LoRA is injected in place: `asr` (same module objects) decodes with it
    if cfg["gradient_checkpointing"]:
        model.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.thinker.enable_input_require_grads()

    root = Path(cfg["data_dir"])
    train_rows = read_manifest(root / "train.jsonl")
    dev_rows = read_manifest(root / "dev.jsonl")
    hold_rows = read_manifest(root / "miami_holdout.jsonl") if (root / "miami_holdout.jsonl").exists() else []
    dit = cfg["dev_in_train"]
    dev_train = [] if dit == "none" else [r for r in dev_rows if dit == "all" or str(r.get("speaker")) == dit[-1]]
    dev_eval = dev_rows if dit in ("none", "all") else [r for r in dev_rows if r not in dev_train]
    train_rows = [r for r in train_rows if r["duration"] <= cfg["max_clip_seconds"]]
    train_rows = quality_filter(train_rows, root / "quality.jsonl", cfg["quality"]) + [dict(r, kind="dev") for r in dev_train]
    vr = random.Random(1)
    turns = [r for r in hold_rows if r.get("kind", "turn") == "turn" and not r.get("nonspeech")]
    wins = [r for r in hold_rows if r.get("kind") == "window"]
    sets = {"holdout_turn": vr.sample(turns, min(len(turns), cfg["val"]["holdout_turn"])),
            "holdout_window": vr.sample(wins, min(len(wins), cfg["val"]["holdout_window"])), "dev": dev_eval}
    weights = torch.tensor(sampling_weights(train_rows, dev_rows, cfg), dtype=torch.double)
    print(f"[qtrain] train rows={len(train_rows)} dev_in_train={dit} | val { {k: len(v) for k, v in sets.items()} }", flush=True)

    if cfg["num_workers"] == "auto":
        cfg["num_workers"] = max(2, min(10, (os.cpu_count() or 4) - 2))
    ds, collate = QwenClips(train_rows, root, cfg), Collate(processor)
    bs, total = cfg["batch_size"], cfg["max_steps"]
    params = [p for p in pm.parameters() if p.requires_grad]
    ema = EMA(params, cfg["ema_decay"]) if cfg["ema_decay"] > 0 else None
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=(0.9, 0.98), fused=cuda)

    def evaluate(step):
        model.eval()
        if ema:
            ema.swap_in(params)
        res = run_eval(asr, sets, root, cfg["eval_batch_size"])
        score = selection_score(res)
        d = out / f"ckpt_step{step}"
        d.mkdir(exist_ok=True)
        torch.save(adapter_state(pm), d / "adapter_weights.pt")
        if ema:
            ema.restore(params)
        model.train()
        print(f"[eval] step {step} score {score:.4f} | " + " | ".join(f"{k} {v:.4f}" for k, v in res.items()), flush=True)
        with open(out / "experiments.jsonl", "a") as f:
            f.write(json.dumps(dict(step=step, score=score, **res, ts=time.time())) + "\n")
        return score, d.name

    best = [evaluate(0)] if not a.dry else []  # zero-shot baseline on the same sets
    step, micro, epoch, t0, stop = 0, 0, 0, time.time(), False
    model.train()
    while not stop:
        g = torch.Generator().manual_seed(cfg["seed"] + epoch)
        order = torch.multinomial(weights, cfg["epoch_samples"], replacement=True, generator=g).tolist()
        dl = DataLoader(torch.utils.data.Subset(ds, order), batch_size=bs, num_workers=cfg["num_workers"], drop_last=True,
                        collate_fn=collate, pin_memory=cuda, prefetch_factor=4 if cfg["num_workers"] else None)
        acc = []
        for batch in dl:
            batch = {k: (v.to(device, dtype) if v.is_floating_point() else v.to(device)) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=cuda):
                loss = model.thinker(**batch).loss
            (loss / cfg["grad_accum"]).backward()
            acc.append(loss.item())
            micro += 1
            if micro % cfg["grad_accum"]:
                continue
            gn = torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            lr = cosine_lr(step, total, cfg["warmup_steps"], cfg["lr"], cfg["min_lr"])
            for gr in opt.param_groups:
                gr["lr"] = lr
            opt.step(); opt.zero_grad(set_to_none=True)
            if ema:
                ema.update(params, step)
            step += 1
            if step % 10 == 0 or step == 1:
                print(f"[qtrain] step {step}/{total} loss {np.mean(acc):.4f} lr {lr:.2e} gn {float(gn):.2f} "
                      f"{(time.time()-t0)/60:.1f}min", flush=True)
                acc = []
            timeout = cfg["max_train_minutes"] and (time.time() - t0) / 60 >= cfg["max_train_minutes"]
            if step % cfg["eval_steps"] == 0 or step >= total or timeout or a.dry:
                best.append(evaluate(step))
                best.sort()
                for _, name in best[cfg["keep_best"]:]:
                    if name != "ckpt_step0":
                        shutil.rmtree(out / name, ignore_errors=True)
                best = best[: cfg["keep_best"]]
            if step >= total or timeout or a.dry:
                stop = True
                break
        epoch += 1

    # finalise: best single vs average of the best K (by the same selection score), then merge -> merged/
    states = [torch.load(out / n / "adapter_weights.pt", map_location="cpu") for _, n in best if (out / n).exists()]
    cands = {"single": states[0]}
    if len(states) > 1:
        cands["avg"] = average_adapters(states)
    model.eval()
    scores = {}
    for name, sd in cands.items():
        pm.load_state_dict(sd, strict=False)
        res = run_eval(asr, sets, root, cfg["eval_batch_size"])
        scores[name] = dict(score=selection_score(res), **res)
        print(f"[final] {name}: " + " | ".join(f"{k} {v:.4f}" for k, v in scores[name].items()), flush=True)
    winner = min(scores, key=lambda k: scores[k]["score"])
    pm.load_state_dict(cands[winner], strict=False)
    torch.save(cands[winner], out / "final_adapter.pt")
    merged = pm.merge_and_unload()
    dest = out / "merged"
    merged.generation_config.temperature = None  # base ships temperature=1e-6 with do_sample=False (invalid on save); its file is copied below
    merged.save_pretrained(dest, safe_serialization=True)
    processor.save_pretrained(dest)
    src = Path(cfg["model"])
    if not src.exists():
        from huggingface_hub import snapshot_download

        src = Path(snapshot_download(cfg["model"], allow_patterns=["*.json", "*.txt"]))
    for f in HF_FILES:  # the base model's own config/tokenizer files: what vLLM/qwen-asr expect
        if (src / f).exists():
            shutil.copy2(src / f, dest / f)
    (out / "final.json").write_text(json.dumps(dict(winner=winner, scores=scores, steps=step, best=best, cfg=cfg), indent=2))
    print(f"[final] {winner} merged -> {dest} (score {scores[winner]['score']:.4f})", flush=True)


if __name__ == "__main__":
    main()
