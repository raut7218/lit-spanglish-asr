"""Build training / validation manifests + 16 kHz FLAC clips.

    python -m lit.prepare_data --miami_dir MIAMI --dev_dir ENSPA_DEV --out_dir OUT

Miami: parse CHAT, merge same-speaker utterances into single-speaker clips (<=29 s, random target
length so lengths resemble voice notes), drop clips that overlap other speakers / unintelligible
speech, split by SPEAKER (a speaker-disjoint hold-out), write FLAC + manifests.
Dev (35 min, WhatsApp-style): converted 1:1 and used ONLY as validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import SR, load_audio
from .chat import Utt, parse_cha, read_participants
from .normalize import norm

MAX_CLIP_S = 29.0
MIN_CLIP_S = 1.0
PAD_S = 0.12
MAX_GAP_S = 1.5
MAX_INTERJECTION_S = 0.8


def _covered(intervals, lo, hi):
    """Total seconds of [lo,hi] covered by the union of intervals (ms)."""
    segs = sorted((max(a, lo), min(b, hi)) for a, b in intervals if b > lo and a < hi)
    total, cur_end = 0, lo
    for a, b in segs:
        a = max(a, cur_end)
        if b > a:
            total += b - a
            cur_end = b
    return total / 1000.0


def build_clips(utts: list[Utt], rng: random.Random, max_overlap: float = 0.15):
    """Group utterances of one conversation into single-speaker clips. Returns list of dicts."""
    utts = sorted(utts, key=lambda u: (u.start_ms, u.end_ms))
    # (speaker, interval) for anything that is NOT transcribed text of the clip's own speaker
    runs, cur = [], []
    for i, u in enumerate(utts):
        empty = not norm(u.text)
        if empty or u.has_unintelligible or u.dur > MAX_CLIP_S:
            if cur:
                runs.append(cur)
            cur = []
            continue
        if cur:
            prev = cur[-1]
            gap = (u.start_ms - prev.end_ms) / 1000.0
            same = u.speaker == prev.speaker
            # utterances of others that start between prev and u
            between = [
                x for x in utts if x.speaker != prev.speaker and prev.start_ms <= x.start_ms < u.end_ms
                and x is not u and x is not prev
            ]
            interjection_only = all(
                x.dur <= MAX_INTERJECTION_S and len(norm(x.text).split()) <= 2 and norm(x.text) for x in between
            )
            if not (same and gap <= MAX_GAP_S and interjection_only):
                runs.append(cur)
                cur = []
        cur.append(u)
    if cur:
        runs.append(cur)

    clips = []
    for run in runs:
        i = 0
        while i < len(run):
            target = rng.uniform(6.0, MAX_CLIP_S)
            j = i
            while j + 1 < len(run) and (run[j + 1].end_ms - run[i].start_ms) / 1000.0 + 2 * PAD_S <= target:
                j += 1
            start_ms = max(0, run[i].start_ms - int(PAD_S * 1000))
            end_ms = run[j].end_ms + int(PAD_S * 1000)
            dur = (end_ms - start_ms) / 1000.0
            if dur > MAX_CLIP_S + 0.5:  # single long utterance; skip
                i = j + 1
                continue
            group = run[i : j + 1]
            text = norm(" ".join(u.text for u in group))
            others = [
                (x.start_ms, x.end_ms)
                for x in utts
                if x.speaker != group[0].speaker and x.end_ms > start_ms and x.start_ms < end_ms
            ]
            overlap = _covered(others, start_ms, end_ms) / max(dur, 1e-6)
            n_words = len(text.split())
            wps = n_words / dur
            n_spa, n_eng = sum(u.n_spa for u in group), sum(u.n_eng for u in group)
            ok = dur >= MIN_CLIP_S and n_words >= 1 and overlap <= max_overlap and 0.3 <= wps <= 6.5
            if ok:
                clips.append(
                    dict(start=start_ms / 1000.0, end=end_ms / 1000.0, duration=dur, text=text,
                         speaker=group[0].speaker, overlap=round(overlap, 3),
                         spa_frac=round(n_spa / max(1, n_spa + n_eng), 3))
                )
            i = j + 1
    return clips


def _process_conv(args):
    cha, mp3, out_dir, seed, max_overlap = args
    conv = Path(cha).stem
    utts = parse_cha(cha)
    rng = random.Random(f"{seed}-{conv}")
    clips = build_clips(utts, rng, max_overlap)
    if not clips:
        return conv, []
    audio = load_audio(mp3, SR)
    rows = []
    (Path(out_dir) / "clips").mkdir(parents=True, exist_ok=True)
    for k, c in enumerate(clips):
        a, b = int(c["start"] * SR), min(len(audio), int(c["end"] * SR))
        if b - a < SR * MIN_CLIP_S:
            continue
        rel = f"clips/{conv}_{k:05d}.flac"
        sf.write(Path(out_dir) / rel, audio[a:b], SR, subtype="PCM_16")
        rows.append(dict(id=f"{conv}_{k:05d}", audio=rel, conv=conv, **{k2: v for k2, v in c.items() if k2 not in ("start", "end")},
                         t0=c["start"], duration=round((b - a) / SR, 3)))
    return conv, rows


def speaker_components(cha_files: list[Path]):
    """Group conversations into speaker-disjoint components (files sharing a speaker are joined)."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for f in cha_files:
        group = "".join(c for c in f.stem if c.isalpha())
        codes = [c for c, role in read_participants(f).items() if "non_participant" not in role]
        find(f.stem)
        for c in codes:
            parent[find(f.stem)] = find(f"{group}:{c}")
    comps = defaultdict(list)
    for f in cha_files:
        comps[find(f.stem)].append(f.stem)
    return list(comps.values())


def choose_holdout(components, durations: dict, hours: float, rng: random.Random):
    comps = sorted(components, key=lambda c: (len(c), sorted(c)))
    rng.shuffle(comps)
    held, total = [], 0.0
    for c in comps:
        d = sum(durations.get(x, 0) for x in c)
        if total + d <= hours * 3600 or not held:
            held.append(c)
            total += d
        if total >= hours * 3600 * 0.8:
            break
    return {x for c in held for x in c}


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def prepare_dev(dev_dir: Path, out_dir: Path):
    rows = []
    (out_dir / "dev_clips").mkdir(parents=True, exist_ok=True)
    meta = list(csv.DictReader(open(dev_dir / "metadata.tsv", encoding="utf-8"), delimiter="\t"))
    for r in meta:
        name = r["audio_filename"]
        audio = load_audio(dev_dir / "clips" / name, SR)
        rel = f"dev_clips/{Path(name).stem}.flac"
        sf.write(out_dir / rel, audio, SR, subtype="PCM_16")
        rows.append(dict(id=Path(name).stem, audio=rel, orig=name, ref=r["transcript"], text=norm(r["transcript"]),
                         duration=round(len(audio) / SR, 3), speaker=r.get("speaker"), conv="dev"))
    write_jsonl(out_dir / "dev.jsonl", rows)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--miami_dir", required=True)
    ap.add_argument("--dev_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--holdout_hours", type=float, default=1.5)
    ap.add_argument("--max_overlap", type=float, default=0.15)
    ap.add_argument("--max_convs", type=int, default=0, help="debug: only first N conversations")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args(argv)

    miami, dev, out = Path(a.miami_dir), Path(a.dev_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    chas = sorted((miami / "chat").glob("*.cha"))
    chas = [c for c in chas if (miami / "audios" / f"{c.stem}.mp3").exists()]
    if a.max_convs:
        chas = chas[: a.max_convs]
    jobs = [(str(c), str(miami / "audios" / f"{c.stem}.mp3"), str(out), a.seed, a.max_overlap) for c in chas]
    print(f"[prepare] {len(jobs)} Miami conversations")
    all_rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for conv, rows in ex.map(_process_conv, jobs):
            print(f"  {conv}: {len(rows)} clips, {sum(r['duration'] for r in rows)/60:.1f} min")
            all_rows += rows

    durations = defaultdict(float)
    for r in all_rows:
        durations[r["conv"]] += r["duration"]
    comps = speaker_components(chas)
    held = choose_holdout(comps, durations, a.holdout_hours, random.Random(a.seed)) if a.holdout_hours > 0 else set()
    train = [r for r in all_rows if r["conv"] not in held]
    hold = [r for r in all_rows if r["conv"] in held]
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "miami_holdout.jsonl", hold)
    dev_rows = prepare_dev(dev, out)

    def h(rows):
        return sum(r["duration"] for r in rows) / 3600

    print(f"[prepare] train {len(train)} clips {h(train):.2f} h | holdout {len(hold)} clips {h(hold):.2f} h "
          f"(convs: {sorted(held)}) | dev {len(dev_rows)} clips {h(dev_rows):.2f} h")
    print(f"[prepare] speaker-disjoint components: {len(comps)}")


if __name__ == "__main__":
    main()
