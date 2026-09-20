"""LoRA fine-tuning of Whisper on Bangor Miami (validated on the 35-min dev set).

    python -m lit.train --config configs/colab_t4.yaml [--set key=value ...]

Resumable (state saved every `save_steps` to `out_dir`, put it on Google Drive), time-boxed
(`max_train_minutes`), logs every eval to `experiments.jsonl`.
"""

from __future__ import annotations

import argparse
from functools import partial
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperFeatureExtractor

from .audio import load_audio
from .augment import AugConfig, Augmenter
from .features import LogMel, pad_batch, spec_augment
from .model_utils import (encode_targets, load_base, load_clip_arrays, load_tokenizer, pick_dtype, read_manifest,
                          transcribe_hf)
from .normalize import wer
from .postprocess import postprocess

DEFAULTS = dict(
    model="openai/whisper-large-v3-turbo",
    language="es",
    data_dir="/content/data_prepared",
    out_dir="/content/drive/MyDrive/lit_runs/run1",
    dtype="auto",  # bf16 if supported, else fp16
    lora=dict(r=32, alpha=64, dropout=0.05, targets=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]),
    lr=1e-4, min_lr=1e-6, warmup_steps=100, weight_decay=1e-3, grad_clip=1.0, label_smoothing=0.1,
    batch_size=4, grad_accum=8, epochs=4, max_steps=0,
    eval_steps=200, save_steps=200, keep_best=3, eval_batch_size=8, eval_beams=1,
    eval_max_clips=0, holdout_eval_clips=60, max_train_minutes=0, num_workers=2,
    gradient_checkpointing=True,
    spec_augment=True, augment=dict(), include_dev_in_train=False, seed=13,
    max_clip_seconds=29.5,
)


def load_config(path, overrides=()):
    cfg = json.loads(json.dumps(DEFAULTS))
    if path:
        user = yaml.safe_load(open(path)) or {}
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    for o in overrides:
        k, v = o.split("=", 1)
        d = cfg
        *path, last = k.split(".")  # dotted keys reach nested dicts: lora.r=64
        for part in path:
            d = d[part]
        d[last] = yaml.safe_load(v)
    return cfg


class ClipDataset(Dataset):
    def __init__(self, rows, root, tok, augmenter, cfg, train=True):
        self.rows, self.root, self.tok, self.aug, self.cfg, self.train = rows, Path(root), tok, augmenter, cfg, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        audio = load_audio(self.root / r["audio"])
        rng = np.random.default_rng((self.cfg["seed"], i, int(time.time() * 1e3) % 100000))
        if self.train and self.aug is not None:
            audio = self.aug(audio, rng)
        audio = audio[: int(self.cfg["max_clip_seconds"] * 16000)]
        dec_in, labels = encode_targets(self.tok, r["text"])
        return audio, dec_in, labels


def collate(batch, pad_id=50257):
    audios, dec_ins, labels = zip(*batch)
    L = max(len(x) for x in dec_ins)
    di = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lb = torch.full((len(batch), L), -100, dtype=torch.long)
    for i, (d, l) in enumerate(zip(dec_ins, labels)):
        di[i, : len(d)] = torch.tensor(d)
        lb[i, : len(l)] = torch.tensor(l)
    lens = [int(len(a) / 160) for a in audios]
    return pad_batch(list(audios)), di, lb, lens


def cosine_lr(step, total, warmup, lr, min_lr):
    if step < warmup:
        return lr * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * p))


def evaluate(model, tok, fe, rows, root, cfg, device, amp_dtype, n_max=0):
    rows = rows[:n_max] if n_max else rows
    audios = load_clip_arrays(rows, root)
    hyps = transcribe_hf(model, tok, fe, audios, device, cfg["language"], cfg["eval_batch_size"], cfg["eval_beams"],
                         amp_dtype=amp_dtype)
    refs = [r.get("ref", r["text"]) for r in rows]
    return wer(refs, [postprocess(h) for h in hyps]), hyps  # score what we would ship (loops collapsed)


def adapter_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}


def average_adapters(states):
    keys = states[0].keys()
    return {k: sum(s[k].float() for s in states) / len(states) for k in keys}


def log_experiment(cfg, record):
    p = Path(cfg["out_dir"]) / "experiments.jsonl"
    with open(p, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = pick_dtype(cfg["dtype"]) if device.type == "cuda" else torch.float32
    use_scaler = device.type == "cuda" and dtype == torch.float16
    amp_dtype = dtype if device.type == "cuda" else None
    print(f"[train] device={device} dtype={dtype} model={cfg['model']}")

    fe = WhisperFeatureExtractor.from_pretrained(cfg["model"])
    tok = load_tokenizer(cfg["model"], cfg["language"])
    model = load_base(cfg["model"], dtype, device)
    lcfg = LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"],
                      target_modules=cfg["lora"]["targets"], bias="none")
    model = get_peft_model(model, lcfg)
    if cfg["gradient_checkpointing"]:
        model.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()
    mel = LogMel(fe, device)

    root = Path(cfg["data_dir"])
    train_rows = read_manifest(root / "train.jsonl")
    dev_rows = read_manifest(root / "dev.jsonl")
    hold_rows = read_manifest(root / "miami_holdout.jsonl") if (root / "miami_holdout.jsonl").exists() else []
    if cfg["include_dev_in_train"]:
        train_rows = train_rows + dev_rows
    train_rows = [r for r in train_rows if r["duration"] <= cfg["max_clip_seconds"]]
    rng_h = random.Random(1)
    hold_eval = rng_h.sample(hold_rows, min(len(hold_rows), cfg["holdout_eval_clips"])) if hold_rows else []
    print(f"[train] train clips={len(train_rows)} ({sum(r['duration'] for r in train_rows)/3600:.1f} h) "
          f"dev={len(dev_rows)} holdout_eval={len(hold_eval)}")

    pool = []
    if cfg["augment"].get("p_babble", 0.15) > 0:
        for r in random.Random(2).sample(train_rows, min(40, len(train_rows))):
            pool.append(load_audio(root / r["audio"])[: 16000 * 12])
    aug = Augmenter(AugConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg["augment"].items()}), pool)
    ds = ClipDataset(train_rows, root, tok, aug, cfg, train=True)
    pad_id = tok.eos_token_id
    steps_per_epoch = max(1, len(ds) // (cfg["batch_size"] * cfg["grad_accum"]))
    total_steps = cfg["max_steps"] or steps_per_epoch * cfg["epochs"]
    print(f"[train] steps/epoch={steps_per_epoch} total_steps={total_steps}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=(0.9, 0.98))
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # ---- resume
    step, epoch, batches_done = 0, 0, 0
    best = []  # list of (dev_wer, step, dirname)
    state_path = out / "train_state.pt"
    if state_path.exists() and (out / "last_adapter").exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        sd = torch.load(out / "last_adapter" / "adapter_weights.pt", map_location="cpu")
        model.load_state_dict(sd, strict=False)
        opt.load_state_dict(st["opt"])
        if use_scaler and st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        step, epoch, batches_done, best = st["step"], st["epoch"], st["batches_done"], st["best"]
        print(f"[train] RESUMED at step {step} (epoch {epoch}, best={best[:1]})")

    t_start = time.time()
    model.train()
    bs = cfg["batch_size"]
    stop = False
    micro = 0
    while step < total_steps and not stop:
        g = torch.Generator().manual_seed(cfg["seed"] + epoch)
        order = torch.randperm(len(ds), generator=g).tolist()
        skip = batches_done * bs
        idxs = order[skip:]
        sub = torch.utils.data.Subset(ds, idxs)
        dl = DataLoader(sub, batch_size=bs, shuffle=False, num_workers=cfg["num_workers"], drop_last=True,
                        collate_fn=partial(collate, pad_id=pad_id), persistent_workers=False, prefetch_factor=4 if cfg["num_workers"] else None)
        acc_loss, acc_n = 0.0, 0
        for wav, dec_in, labels, lens in dl:
            feats = mel(wav)
            if cfg["spec_augment"]:
                feats = spec_augment(feats, lens)
            feats = feats.to(dtype)
            dec_in, labels = dec_in.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(input_features=feats, decoder_input_ids=dec_in).logits
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100,
                                   label_smoothing=cfg["label_smoothing"])
            scaler.scale(loss / cfg["grad_accum"]).backward()
            acc_loss += loss.item(); acc_n += 1; micro += 1; batches_done += 1
            if micro % cfg["grad_accum"] != 0:
                continue
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            lr = cosine_lr(step, total_steps, cfg["warmup_steps"], cfg["lr"], cfg["min_lr"])
            for gr in opt.param_groups:
                gr["lr"] = lr
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
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
                w_dev, _ = evaluate(model, tok, fe, dev_rows, root, cfg, device, amp_dtype, cfg["eval_max_clips"])
                w_hold = evaluate(model, tok, fe, hold_eval, root, cfg, device, amp_dtype)[0] if hold_eval else float("nan")
                model.train()
                print(f"[eval] step {step} dev WER {w_dev:.4f} | miami-holdout WER {w_hold:.4f}", flush=True)
                log_experiment(cfg, dict(step=step, dev_wer=w_dev, holdout_wer=w_hold, train_hours_seen=step * bs * cfg["grad_accum"] * 15 / 3600, ts=time.time()))
                d = out / f"ckpt_step{step}"
                d.mkdir(exist_ok=True)
                torch.save(adapter_state(model), d / "adapter_weights.pt")
                best.append((w_dev, step, d.name))
                best.sort()
                for _, _, name in best[cfg["keep_best"]:]:
                    shutil.rmtree(out / name, ignore_errors=True)
                best = best[: cfg["keep_best"]]
            if do_save:
                (out / "last_adapter").mkdir(exist_ok=True)
                torch.save(adapter_state(model), out / "last_adapter" / "adapter_weights.pt")
                torch.save(dict(opt=opt.state_dict(), scaler=scaler.state_dict() if use_scaler else None, step=step,
                                epoch=epoch, batches_done=batches_done, best=best), state_path)
            if a.dry or timeout or step >= total_steps:
                stop = True
                break
        else:
            epoch += 1
            batches_done = 0

    # ---- finalise: average best checkpoints, compare with the best single one, save the winner
    states = [torch.load(out / n / "adapter_weights.pt", map_location="cpu") for _, _, n in best if (out / n).exists()]
    final = {"single": states[0]} if states else {"single": adapter_state(model)}
    if len(states) > 1:
        final["avg"] = average_adapters(states)
    scores = {}
    for name, sd in final.items():
        model.load_state_dict(sd, strict=False)
        model.eval()
        scores[name] = evaluate(model, tok, fe, dev_rows, root, cfg, device, amp_dtype, cfg["eval_max_clips"])[0]
        print(f"[final] {name}: dev WER {scores[name]:.4f}")
    winner = min(scores, key=scores.get)
    model.load_state_dict(final[winner], strict=False)
    dest = out / "final_adapter"
    model.save_pretrained(dest)
    (dest / "final.json").write_text(json.dumps(dict(winner=winner, scores=scores, steps=step, cfg=cfg), indent=2))
    log_experiment(cfg, dict(final=True, winner=winner, scores=scores, steps=step, ts=time.time()))
    print(f"[final] saved {winner} adapter -> {dest} (dev WER {scores[winner]:.4f})")


if __name__ == "__main__":
    main()
