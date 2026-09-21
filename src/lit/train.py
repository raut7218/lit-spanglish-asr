"""LoRA fine-tuning of Whisper for Spanglish voice notes (v2: robust + fast).

    python -m lit.train --config configs/colab_a100_v2.yaml [--set key=value ...]

What is different from v1 (see the plan / README):
  * LoRA only where it matters: encoder layers >= `lora.encoder_from_layer` + all decoder layers; the frozen lower
    encoder needs no gradients or stored activations -> ~30% less compute, room for a bigger batch, no checkpointing.
  * Data mix: single-speaker turns + long two-speaker conversation windows + empty-target noise clips (+ optionally
    dev speakers), re-weighted so the Spanish/English share matches the dev/test voice notes.
  * GPU-side augmentation (lit.gpu_aug): level normalisation, denoise, EQ tilt, light noise/reverb; no CPU ffmpeg.
  * EMA weights (with warm-up) are what we validate and export; selection is dev-weighted (cfg `select`), Miami hold-out is the guard.
  * Resumable (state every `save_steps`), time-boxed (`max_train_minutes`), logs to `experiments.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperFeatureExtractor

from .audio import load_audio
from .augment import codec_chain
from .features import LogMel, pad_batch, spec_augment
from .gpu_aug import GpuAugConfig, GpuAugmenter, normalize_level_db
from .model_utils import (encode_targets, load_base, load_clip_arrays, load_tokenizer, pick_dtype, read_manifest,
                          transcribe_hf)
from .normalize import wer
from .postprocess import postprocess

DEFAULTS = dict(
    model="openai/whisper-large-v3",
    language="es",
    data_dir="/content/data_prepared",
    out_dir="/content/drive/MyDrive/lit_runs/run2",
    dtype="auto",  # bf16 if supported, else fp16
    lora=dict(r=64, alpha=64, dropout=0.0, targets=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
              encoder_from_layer=16, decoder=True),
    lr=1e-4, min_lr=1e-6, warmup_steps=50, weight_decay=1e-3, grad_clip=1.0, label_smoothing=0.1,
    batch_size=8, grad_accum=4, epochs=0, max_steps=1200, epoch_samples=12800,
    eval_steps=200, save_steps=200, keep_best=3, eval_batch_size=32, eval_beams=1,
    val=dict(holdout_turn=120, holdout_window=60),
    sampler=dict(kind_share=dict(turn=0.35, window=0.60, nonspeech=0.015), dev_share=0.08, match_dev_language=True),
    dev_in_train="none",  # none | spk1 | spk2 | all   (spk1: train on dev speaker 1, validate on speaker 2 ...)
    ema_decay=0.999,
    select=dict(dev=0.7, holdout_turn=0.3),  # weights of the checkpoint-selection score
    init_adapter="",  # path to an adapter_weights.pt: warm start (weights only; fresh optimiser/schedule/EMA), e.g. stage-2 fine-tune
    max_train_minutes=0, num_workers="auto",
    gradient_checkpointing="auto",  # True | False | "auto" (off, switch on if the GPU runs out of memory)
    cpu_aug=dict(p_speed=0.4, speeds=[0.9, 1.0, 1.1], p_variant=0.9, p_codec=0.3),  # speed, codec-bank pick, online Opus->MP3 chain (no bank)
    gpu_aug=dict(), spec_augment=True, seed=13, max_clip_seconds=29.5,
)

SPA_BINS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001]


def load_config(path, overrides=()):
    cfg = json.loads(json.dumps(DEFAULTS))

    def merge(dst, src):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                merge(dst[k], v)
            else:
                dst[k] = v

    if path:
        merge(cfg, yaml.safe_load(open(path)) or {})
    for o in overrides:
        k, v = o.split("=", 1)
        d = cfg
        *parts, last = k.split(".")  # dotted keys reach nested dicts: lora.r=64
        for part in parts:
            d = d[part]
        d[last] = yaml.safe_load(v)
    return cfg


# --------------------------------------------------------------------------------------- data
class ClipDataset(Dataset):
    def __init__(self, rows, root, tok, cfg, train=True):
        self.rows, self.root, self.tok, self.cfg, self.train = rows, Path(root), tok, cfg, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        ca = self.cfg["cpu_aug"]
        rng = np.random.default_rng((self.cfg["seed"], i, int(time.time() * 1e3) % 100000))
        rel = r["audio"]
        if self.train and r.get("variants") and rng.random() < ca["p_variant"]:
            rel = r["variants"][int(rng.integers(len(r["variants"])))]
        audio = load_audio(self.root / rel)
        if self.train and rng.random() < ca["p_speed"]:
            f = float(rng.choice(ca["speeds"]))
            if abs(f - 1.0) > 1e-3:
                n = int(round(len(audio) / f))
                audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
        audio = audio[: int(self.cfg["max_clip_seconds"] * 16000)]
        if self.train and not r.get("variants") and rng.random() < ca.get("p_codec", 0.0) and not r.get("nonspeech"):
            audio = codec_chain(audio, rng)  # test clips are MP3-64k re-encodes of Opus voice notes
        dec_in, labels = encode_targets(self.tok, r["text"])
        return audio, dec_in, labels, bool(r.get("nonspeech"))


def collate(batch, pad_id=50257):
    audios, dec_ins, labels, ns = zip(*batch)
    L = max(len(x) for x in dec_ins)
    di = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lb = torch.full((len(batch), L), -100, dtype=torch.long)
    for i, (d, l) in enumerate(zip(dec_ins, labels)):
        di[i, : len(d)] = torch.tensor(d)
        lb[i, : len(l)] = torch.tensor(l)
    lens = torch.tensor([len(a) for a in audios], dtype=torch.long)
    return pad_batch(list(audios)), di, lb, lens, torch.tensor(ns)


def spa_bin(x):
    return next(i for i in range(len(SPA_BINS) - 1) if SPA_BINS[i] <= x < SPA_BINS[i + 1])


def sampling_weights(rows, dev_rows, cfg):
    """Per-row sampling weights: fixed share per clip kind, and within a kind the Spanish-share histogram is
    re-weighted to match the dev voice notes (Miami is English-heavy, the test is Spanish-heavy code-switching)."""
    sc = cfg["sampler"]
    share = dict(sc["kind_share"])
    if any(r.get("kind") == "dev" for r in rows):
        share["dev"] = sc["dev_share"]
    dev_h = np.ones(len(SPA_BINS) - 1)
    if sc.get("match_dev_language"):
        vals = [r["spa_frac"] for r in dev_rows if r.get("spa_frac") is not None]
        if vals:
            dev_h = np.bincount([spa_bin(v) for v in vals], minlength=len(dev_h)) / len(vals) + 0.02
    w = np.zeros(len(rows))
    kinds = sorted({r.get("kind", "turn") for r in rows})
    for k in kinds:
        idx = [i for i, r in enumerate(rows) if r.get("kind", "turn") == k]
        base = np.ones(len(idx))
        if k in ("turn", "window") and sc.get("match_dev_language"):
            bins = np.array([spa_bin(rows[i]["spa_frac"]) if rows[i].get("spa_frac") is not None else 0 for i in idx])
            tr_h = np.bincount(bins, minlength=len(dev_h)) / max(1, len(idx)) + 0.02
            base = np.clip(dev_h[bins] / tr_h[bins], 0.25, 4.0)
        w[idx] = base / base.sum() * share.get(k, 0.05)
    return w / w.sum()


# --------------------------------------------------------------------------------------- model helpers
def lora_target_regex(cfg, n_enc: int):
    lc = cfg["lora"]
    if lc.get("encoder_from_layer", 0) <= 0 and lc.get("decoder", True):
        return lc["targets"]
    attn = [t for t in lc["targets"] if t.endswith("_proj")]
    mlp = [t for t in lc["targets"] if t in ("fc1", "fc2")]
    parts = []
    enc = "|".join(str(i) for i in range(lc.get("encoder_from_layer", 0), n_enc))
    if enc:
        parts.append(rf"model\.encoder\.layers\.({enc})\.(self_attn\.({'|'.join(attn)})|{'|'.join(mlp)})")
    if lc.get("decoder", True):
        parts.append(rf"model\.decoder\.layers\.\d+\.((self_attn|encoder_attn)\.({'|'.join(attn)})|{'|'.join(mlp)})")
    return "|".join(parts)


def cosine_lr(step, total, warmup, lr, min_lr):
    if step < warmup:
        return lr * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * p))


class EMA:
    """Exponential moving average with warm-up: decay_t = min(decay, (1+t)/(10+t)). Without it the shadow (initialised
    at the LoRA init, B=0) keeps decay**steps of the untrained weights: 0.999**1000 = 37% of the exported model."""

    def __init__(self, params, decay):
        self.decay, self.n = decay, 0
        self.shadow = [p.detach().clone().float() for p in params]

    def current_decay(self):
        return min(self.decay, (1 + self.n) / (10 + self.n))

    @torch.no_grad()
    def update(self, params):
        self.n += 1
        d = self.current_decay()
        for s, p in zip(self.shadow, params):
            s.mul_(d).add_(p.detach().float(), alpha=1 - d)

    @torch.no_grad()
    def swap_in(self, params):
        self._backup = [p.detach().clone() for p in params]
        for s, p in zip(self.shadow, params):
            p.copy_(s.to(p.dtype))

    @torch.no_grad()
    def restore(self, params):
        for b, p in zip(self._backup, params):
            p.copy_(b)
        self._backup = None


def adapter_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}


def average_adapters(states):
    return {k: sum(s[k].float() for s in states) / len(states) for k in states[0]}


def log_experiment(cfg, record):
    with open(Path(cfg["out_dir"]) / "experiments.jsonl", "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_eval(model, tok, fe, sets, root, cfg, device, amp_dtype):
    """{name: rows} -> {name: WER} on the shipped-style text (loops collapsed, scorer normalised)."""
    out = {}
    for name, rows in sets.items():
        if not rows:
            continue
        hyps = transcribe_hf(model, tok, fe, load_clip_arrays(rows, root), device, cfg["language"], cfg["eval_batch_size"],
                             cfg["eval_beams"], amp_dtype=amp_dtype)
        out[name] = wer([r.get("ref", r["text"]) for r in rows], [postprocess(h) for h in hyps])
    return out


def selection_score(res: dict, weights: dict | None = None) -> float:
    """Weighted average of the Miami hold-out (turns) and the dev voice notes (the test is dev-like, so dev leads)."""
    w = weights or DEFAULTS["select"]
    keys = [k for k in w if k in res]
    return float(sum(w[k] * res[k] for k in keys) / sum(w[k] for k in keys)) if keys else float("nan")


# --------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--dry", action="store_true", help="build everything, run 1 step, exit")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = pick_dtype(cfg["dtype"]) if device.type == "cuda" else torch.float32
    use_scaler = device.type == "cuda" and dtype == torch.float16
    amp_dtype = dtype if device.type == "cuda" else None
    print(f"[train] device={device} dtype={dtype} model={cfg['model']}")

    fe = WhisperFeatureExtractor.from_pretrained(cfg["model"])
    tok = load_tokenizer(cfg["model"], cfg["language"])
    model = load_base(cfg["model"], dtype, device)
    n_enc = model.config.encoder_layers
    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules=lora_target_regex(cfg, n_enc), bias="none"))
    if cfg["init_adapter"]:
        sd0 = torch.load(cfg["init_adapter"], map_location="cpu")
        miss = model.load_state_dict(sd0, strict=False)
        assert not miss.unexpected_keys, f"init_adapter does not match this LoRA layout: {miss.unexpected_keys[:3]}"
        print(f"[train] warm start from {cfg['init_adapter']} ({len(sd0)} tensors)")
    model.print_trainable_parameters()
    n_lora = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A"))
    print(f"[train] LoRA modules: {n_lora} (encoder layers >= {lc.get('encoder_from_layer', 0)}, decoder={lc.get('decoder', True)})")

    gc_on = {"v": False}

    def enable_gc():
        model.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        gc_on["v"] = True
        print("[train] gradient checkpointing ON", flush=True)

    if cfg["gradient_checkpointing"] is True:
        enable_gc()
    mel = LogMel(fe, device)
    gpu_aug = GpuAugmenter(GpuAugConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg["gpu_aug"].items()}))

    # ---- data
    root = Path(cfg["data_dir"])
    bank = root / "train_bank.jsonl"
    train_rows = read_manifest(bank if bank.exists() else root / "train.jsonl")
    dev_rows = read_manifest(root / "dev.jsonl")
    hold_rows = read_manifest(root / "miami_holdout.jsonl") if (root / "miami_holdout.jsonl").exists() else []
    dit = cfg["dev_in_train"]
    dev_train = [] if dit == "none" else [r for r in dev_rows if dit == "all" or str(r.get("speaker")) == dit[-1]]
    dev_eval = dev_rows if dit in ("none", "all") else [r for r in dev_rows if r not in dev_train]
    sel_w = cfg["select"]
    if dit == "all":
        print("[train] WARNING: dev is in the training set; 'dev' WER is contaminated, selecting on the hold-out only")
        sel_w = {"holdout_turn": 1.0}
    dev_train = [dict(r, kind="dev") for r in dev_train]
    train_rows = [r for r in train_rows if r["duration"] <= cfg["max_clip_seconds"]] + dev_train
    val_rng = random.Random(1)
    turns = [r for r in hold_rows if r.get("kind", "turn") == "turn"]
    wins = [r for r in hold_rows if r.get("kind") == "window"]
    sets = {
        "holdout_turn": val_rng.sample(turns, min(len(turns), cfg["val"]["holdout_turn"])),
        "holdout_window": val_rng.sample(wins, min(len(wins), cfg["val"]["holdout_window"])),
        "dev": dev_eval,
    }
    weights = torch.tensor(sampling_weights(train_rows, dev_rows, cfg), dtype=torch.double)
    kinds = {k: sum(1 for r in train_rows if r.get("kind", "turn") == k) for k in {r.get("kind", "turn") for r in train_rows}}
    print(f"[train] train rows={len(train_rows)} kinds={kinds} dev_in_train={dit} | val sizes "
          f"{ {k: len(v) for k, v in sets.items()} }")

    if cfg["num_workers"] == "auto":
        cfg["num_workers"] = max(2, min(10, (os.cpu_count() or 4) - 2))
    print(f"[train] dataloader workers={cfg['num_workers']}")
    ds = ClipDataset(train_rows, root, tok, cfg, train=True)
    pad_id = tok.eos_token_id
    bs = cfg["batch_size"]
    steps_per_epoch = max(1, cfg["epoch_samples"] // (bs * cfg["grad_accum"]))
    total_steps = cfg["max_steps"] or steps_per_epoch * max(1, cfg["epochs"])
    print(f"[train] sampler epoch={steps_per_epoch} steps, total_steps={total_steps}, effective batch={bs * cfg['grad_accum']}")

    params = [p for p in model.parameters() if p.requires_grad]
    ema = EMA(params, cfg["ema_decay"]) if cfg["ema_decay"] > 0 else None
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=(0.9, 0.98), fused=device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # ---- resume
    step, epoch, batches_done, best = 0, 0, 0, []
    state_path = out / "train_state.pt"
    if state_path.exists() and (out / "last_adapter").exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(torch.load(out / "last_adapter" / "adapter_weights.pt", map_location="cpu"), strict=False)
        opt.load_state_dict(st["opt"])
        if use_scaler and st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        if ema and st.get("ema"):
            ema.shadow = [t.to(device) for t in st["ema"]]
        step, epoch, batches_done, best = st["step"], st["epoch"], st["batches_done"], st["best"]
        if ema:
            ema.n = step
        print(f"[train] RESUMED at step {step} (epoch {epoch}, best={best[:1]})")

    t_start = time.time()
    model.train()
    stop, micro = False, 0
    while step < total_steps and not stop:
        g = torch.Generator().manual_seed(cfg["seed"] + epoch)
        order = torch.multinomial(weights, cfg["epoch_samples"], replacement=True, generator=g).tolist()
        sub = torch.utils.data.Subset(ds, order[batches_done * bs:])
        dl = DataLoader(sub, batch_size=bs, shuffle=False, num_workers=cfg["num_workers"], drop_last=True,
                        collate_fn=partial(collate, pad_id=pad_id), pin_memory=device.type == "cuda",
                        prefetch_factor=6 if cfg["num_workers"] else None)
        acc_loss, acc_n = 0.0, 0
        for wav, dec_in, labels, lens, is_ns in dl:
            wav, lens = wav.to(device, non_blocking=True), lens.to(device)
            wav = gpu_aug(wav, lens)
            if is_ns.any():  # real silence/noise clips are quiet: undo the loudness normalisation for them
                m = is_ns.to(device)
                wav[m] = normalize_level_db(wav[m], lens[m], -30.0 - 20.0 * torch.rand(int(m.sum()), device=device))
            feats = mel(wav)
            if cfg["spec_augment"]:
                feats = spec_augment(feats, (lens // 160).tolist())
            feats = feats.to(dtype)
            dec_in, labels = dec_in.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            for attempt in range(2):
                try:
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                        logits = model(input_features=feats, decoder_input_ids=dec_in).logits
                    loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100,
                                           label_smoothing=cfg["label_smoothing"])
                    scaler.scale(loss / cfg["grad_accum"]).backward()
                    break
                except torch.cuda.OutOfMemoryError:
                    if gc_on["v"] or attempt == 1:
                        raise
                    logits = loss = None
                    opt.zero_grad(set_to_none=True)  # drop the partial accumulation window; restart it with this batch
                    micro -= micro % cfg["grad_accum"]
                    torch.cuda.empty_cache()
                    print("[train] OOM without checkpointing -> enabling gradient checkpointing and retrying", flush=True)
                    enable_gc()
            acc_loss += loss.item(); acc_n += 1; micro += 1; batches_done += 1
            if micro % cfg["grad_accum"] != 0:
                continue
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            lr = cosine_lr(step, total_steps, cfg["warmup_steps"], cfg["lr"], cfg["min_lr"])
            for gr in opt.param_groups:
                gr["lr"] = lr
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            if ema:
                ema.update(params)
            step += 1
            if step % 10 == 0 or step == 1:
                el = (time.time() - t_start) / 60
                print(f"[train] step {step}/{total_steps} loss {acc_loss/acc_n:.4f} lr {lr:.2e} gn {float(gn):.2f} {el:.1f}min", flush=True)
                acc_loss, acc_n = 0.0, 0

            timeout = cfg["max_train_minutes"] and (time.time() - t_start) / 60 >= cfg["max_train_minutes"]
            do_eval = step % cfg["eval_steps"] == 0 or step == total_steps or timeout or a.dry
            do_save = step % cfg["save_steps"] == 0 or step == total_steps or timeout or a.dry
            if do_eval:
                model.eval()
                if ema:
                    ema.swap_in(params)
                res = run_eval(model, tok, fe, sets, root, cfg, device, amp_dtype)
                score = selection_score(res, sel_w)
                d = out / f"ckpt_step{step}"
                d.mkdir(exist_ok=True)
                torch.save(adapter_state(model), d / "adapter_weights.pt")  # EMA weights (what we select/export)
                if ema:
                    ema.restore(params)
                model.train()
                print(f"[eval] step {step} score {score:.4f} | " + " | ".join(f"{k} {v:.4f}" for k, v in res.items()), flush=True)
                log_experiment(cfg, dict(step=step, score=score, **res, ts=time.time()))
                best.append((score, step, d.name))
                best.sort()
                for _, _, name in best[cfg["keep_best"]:]:
                    shutil.rmtree(out / name, ignore_errors=True)
                best = best[: cfg["keep_best"]]
            if do_save:
                (out / "last_adapter").mkdir(exist_ok=True)
                torch.save(adapter_state(model), out / "last_adapter" / "adapter_weights.pt")
                torch.save(dict(opt=opt.state_dict(), scaler=scaler.state_dict() if use_scaler else None, step=step,
                                epoch=epoch, batches_done=batches_done, best=best,
                                ema=[t.cpu() for t in ema.shadow] if ema else None), state_path)
            if a.dry or timeout or step >= total_steps:
                stop = True
                break
        else:
            epoch += 1
            batches_done = 0

    # ---- finalise: best single EMA checkpoint vs average of the best K; keep the winner on the selection score
    states = [torch.load(out / n / "adapter_weights.pt", map_location="cpu") for _, _, n in best if (out / n).exists()]
    final = {"single": states[0]} if states else {"single": adapter_state(model)}
    if len(states) > 1:
        final["avg"] = average_adapters(states)
    scores = {}
    model.eval()
    for name, sd in final.items():
        model.load_state_dict(sd, strict=False)
        res = run_eval(model, tok, fe, sets, root, cfg, device, amp_dtype)
        scores[name] = dict(score=selection_score(res, sel_w), **res)
        print(f"[final] {name}: " + " | ".join(f"{k} {v:.4f}" for k, v in scores[name].items()))
    winner = min(scores, key=lambda k: scores[k]["score"])
    model.load_state_dict(final[winner], strict=False)
    dest = out / "final_adapter"
    model.save_pretrained(dest)
    (dest / "final.json").write_text(json.dumps(dict(winner=winner, scores=scores, steps=step, cfg=cfg), indent=2))
    log_experiment(cfg, dict(final=True, winner=winner, scores=scores, steps=step, ts=time.time()))
    print(f"[final] saved {winner} adapter -> {dest} (score {scores[winner]['score']:.4f})")


if __name__ == "__main__":
    main()
