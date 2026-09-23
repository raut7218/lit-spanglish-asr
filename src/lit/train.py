"""Full fine-tuning of Canary-1b-v2 (lit.canary, NeMo-free) for Spanglish voice notes.

    python -m lit.train --config configs/canary_a100.yaml [--set key=value ...]

  * `model` is a dir made by scripts/convert_canary.py. Encoder layers < `freeze_encoder_below` (and the conv
    subsampling) are frozen: no gradients, no stored activations. Everything above trains, encoder at `lr_encoder`,
    decoder at `lr`. BatchNorm statistics stay frozen (lit.canary.CanaryModel.train).
  * Prompt per clip: language token from the clip's Spanish share (es/en) + the `<|verbatim|>` style token, so the
    model learns verbatim transcription (disfluencies, repetitions) as a mode separate from its clean pre-training.
  * Data mix: single-speaker turns + long two-speaker conversation windows + empty-target noise clips (+ optionally
    dev speakers), re-weighted so the Spanish/English share matches the dev/test voice notes.
  * GPU-side augmentation (lit.gpu_aug): level normalisation, denoise, EQ tilt, light noise/reverb; no CPU ffmpeg.
  * EMA weights are what we validate and export; selection uses macro(Miami hold-out, dev) - never dev alone.
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
import yaml
from torch.utils.data import DataLoader, Dataset

from . import canary
from .audio import load_audio, load_clip_arrays, read_manifest
from .augment import codec_chain
from .gpu_aug import GpuAugConfig, GpuAugmenter, normalize_level_db, spec_augment
from .normalize import wer
from .postprocess import postprocess

DEFAULTS = dict(
    model="/content/canary_base",  # scripts/convert_canary.py output
    languages=["es", "en"],  # eval/inference prompts; the best-scoring hypothesis wins
    verbatim=True,  # append the <|verbatim|> style token to every prompt
    data_dir="/content/data_prepared",
    out_dir="/content/drive/MyDrive/lit_runs/canary1",
    dtype="auto",  # autocast dtype: bf16 if supported, else fp16 (master weights stay fp32)
    freeze_encoder_below=16,
    lr=1e-4, lr_encoder=3e-5, min_lr=1e-6, warmup_steps=200, weight_decay=1e-3, grad_clip=1.0, label_smoothing=0.1,
    batch_size=8, grad_accum=4, epochs=0, max_steps=2500, epoch_samples=12800,
    eval_steps=200, save_steps=200, keep_best=3, eval_batch_size=32, eval_beams=1,
    val=dict(holdout_turn=120, holdout_window=60),
    sampler=dict(kind_share=dict(turn=0.35, window=0.60, nonspeech=0.015), dev_share=0.08, match_dev_language=True),
    dev_in_train="none",  # none | spk1 | spk2 | all   (spk1: train on dev speaker 1, validate on speaker 2 ...)
    ema_decay=0.9995,
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
        *parts, last = k.split(".")  # dotted keys reach nested dicts: val.holdout_turn=60
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
        dec_in, labels = encode_targets(self.tok, r["text"], clip_language(r))
        return audio, dec_in, labels, bool(r.get("nonspeech"))


def clip_language(r) -> str:
    """Prompt language of a training clip: its dominant language (Canary's src/tgt token); dev/unknown -> es."""
    f = r.get("spa_frac")
    return "en" if f is not None and f < 0.5 else "es"


def encode_targets(tok, text: str, lang: str, max_len: int = 440):
    """prompt + text + eos -> (decoder input, labels); the prompt positions are not scored."""
    prompt = tok.prompt(lang)
    ids = (prompt + tok.encode(text))[: max_len - 1] + [tok.eos]
    labels = [-100] * (len(prompt) - 1) + ids[len(prompt):]
    return ids[:-1], labels


def collate(batch, pad_id=2):
    audios, dec_ins, labels, ns = zip(*batch)
    L = max(len(x) for x in dec_ins)
    di = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lb = torch.full((len(batch), L), -100, dtype=torch.long)
    for i, (d, l) in enumerate(zip(dec_ins, labels)):
        di[i, : len(d)] = torch.tensor(d)
        lb[i, : len(l)] = torch.tensor(l)
    wav, lens = canary.pad_waves(list(audios))
    return wav, di, lb, lens, torch.tensor(ns)


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
def cosine_lr(step, total, warmup, lr, min_lr):
    if step < warmup:
        return lr * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * p))


class EMA:
    def __init__(self, params, decay):
        self.decay = decay
        self.shadow = [p.detach().clone().float() for p in params]

    @torch.no_grad()
    def update(self, params, step: int):
        # warmup: a flat 0.9995 kept ~29% of the *base* weights in the EMA after 2500 steps (0.9995**2500)
        d = min(self.decay, (1 + step) / (10 + step))
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


def trainable_state(model, dtype=None):
    """Only the trained tensors (the frozen part never changes): what checkpoints and resume need."""
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().to("cpu", dtype=dtype or v.dtype).clone() for k, v in model.state_dict().items() if k in names}


def load_partial(model, sd):
    unknown = set(sd) - set(model.state_dict())
    assert not unknown, f"unknown keys {sorted(unknown)[:5]}"
    model.load_state_dict(sd, strict=False)


def average_states(states):
    return {k: sum(s[k].float() for s in states) / len(states) for k in states[0]}


def log_experiment(cfg, record):
    with open(Path(cfg["out_dir"]) / "experiments.jsonl", "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_eval(model, tok, sets, root, cfg, amp_dtype):
    """{name: rows} -> {name: WER} on the shipped-style text (loops collapsed, scorer normalised)."""
    out = {}
    for name, rows in sets.items():
        if not rows:
            continue
        hyps = canary.transcribe(model, tok, load_clip_arrays(rows, root), cfg["languages"], cfg["eval_beams"],
                                 cfg["eval_batch_size"], amp_dtype=amp_dtype)
        out[name] = wer([r.get("ref", r["text"]) for r in rows], [postprocess(h) for h in hyps])
    return out


def selection_score(res: dict) -> float:
    """Macro average of the multi-speaker Miami hold-out (turns) and the dev voice notes."""
    keys = [k for k in ("holdout_turn", "dev") if k in res]
    return float(np.mean([res[k] for k in keys])) if keys else float("nan")


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
    dtype = canary.pick_dtype(cfg["dtype"]) if device.type == "cuda" else torch.float32
    use_scaler = device.type == "cuda" and dtype == torch.float16
    amp_dtype = dtype if device.type == "cuda" else None
    print(f"[train] device={device} autocast={dtype} model={cfg['model']}")

    model, tok = canary.load(cfg["model"], device, torch.float32)  # fp32 master weights, bf16/fp16 autocast
    if not cfg["verbatim"]:
        tok.verbatim = model.cfg["verbatim_token"] = None
    model.freeze_encoder_below(cfg["freeze_encoder_below"])
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[train] trainable {n_tr/1e6:.0f}M / {n_all/1e6:.0f}M (encoder layers >= {cfg['freeze_encoder_below']} + decoder), "
          f"verbatim token={tok.verbatim}")

    gc_on = {"v": False}

    def enable_gc():
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        gc_on["v"] = True
        print("[train] gradient checkpointing ON (encoder)", flush=True)

    if cfg["gradient_checkpointing"] is True:
        enable_gc()
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
    if dit == "all":
        print("[train] WARNING: dev is in the training set; 'dev' WER is contaminated, use the hold-out")
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
    pad_id = tok.pad
    bs = cfg["batch_size"]
    steps_per_epoch = max(1, cfg["epoch_samples"] // (bs * cfg["grad_accum"]))
    total_steps = cfg["max_steps"] or steps_per_epoch * max(1, cfg["epochs"])
    print(f"[train] sampler epoch={steps_per_epoch} steps, total_steps={total_steps}, effective batch={bs * cfg['grad_accum']}")

    params = [p for p in model.parameters() if p.requires_grad]
    enc_ids = {id(p) for p in model.encoder.parameters()}
    groups = [dict(params=[p for p in params if id(p) in enc_ids], base_lr=cfg["lr_encoder"]),
              dict(params=[p for p in params if id(p) not in enc_ids], base_lr=cfg["lr"])]
    ema = EMA(params, cfg["ema_decay"]) if cfg["ema_decay"] > 0 else None
    opt = torch.optim.AdamW([g for g in groups if g["params"]], lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                            betas=(0.9, 0.98), fused=device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # ---- resume
    step, epoch, batches_done, best = 0, 0, 0, []
    state_path = out / "train_state.pt"
    if state_path.exists() and (out / "last" / "weights.pt").exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        load_partial(model, torch.load(out / "last" / "weights.pt", map_location="cpu"))
        opt.load_state_dict(st["opt"])
        if use_scaler and st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        if ema and st.get("ema"):
            ema.shadow = [t.to(device) for t in st["ema"]]
        step, epoch, batches_done, best = st["step"], st["epoch"], st["batches_done"], st["best"]
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
            feats, fmask = model.features(wav, lens)
            if cfg["spec_augment"]:
                feats = spec_augment(feats.transpose(1, 2), fmask.sum(1).tolist()).transpose(1, 2)
            dec_in, labels = dec_in.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            for attempt in range(2):
                try:
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                        loss = model(feats, fmask, dec_in, labels, cfg["label_smoothing"])
                    scaler.scale(loss / cfg["grad_accum"]).backward()
                    break
                except torch.cuda.OutOfMemoryError:
                    if gc_on["v"] or attempt == 1:
                        raise
                    loss = None
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
                gr["lr"] = lr * gr["base_lr"] / cfg["lr"]
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            if ema:
                ema.update(params, step)
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
                res = run_eval(model, tok, sets, root, cfg, amp_dtype)
                score = selection_score(res)
                d = out / f"ckpt_step{step}"
                d.mkdir(exist_ok=True)
                torch.save(trainable_state(model, torch.float16), d / "weights.pt")  # EMA weights (what we select/export)
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
                (out / "last").mkdir(exist_ok=True)
                torch.save(trainable_state(model), out / "last" / "weights.pt")
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
    states = [torch.load(out / n / "weights.pt", map_location="cpu") for _, _, n in best if (out / n).exists()]
    final = {"single": states[0]} if states else {"single": trainable_state(model)}
    if len(states) > 1:
        final["avg"] = average_states(states)
    scores = {}
    model.eval()
    for name, sd in final.items():
        load_partial(model, sd)
        res = run_eval(model, tok, sets, root, cfg, amp_dtype)
        scores[name] = dict(score=selection_score(res), **res)
        print(f"[final] {name}: " + " | ".join(f"{k} {v:.4f}" for k, v in scores[name].items()))
    winner = min(scores, key=lambda k: scores[k]["score"])
    load_partial(model, final[winner])
    dest = out / "final_model"
    canary.save(model, dest, tokenizer_path=Path(cfg["model"]) / "tokenizer.model", dtype=torch.float16)
    (dest / "final.json").write_text(json.dumps(dict(winner=winner, scores=scores, steps=step, cfg=cfg), indent=2))
    log_experiment(cfg, dict(final=True, winner=winner, scores=scores, steps=step, ts=time.time()))
    print(f"[final] saved {winner} model -> {dest} (score {scores[winner]['score']:.4f})")


if __name__ == "__main__":
    main()
