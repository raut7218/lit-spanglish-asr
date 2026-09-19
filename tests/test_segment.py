import random

from lit.chat import Utt
from lit.prepare_data import MAX_CLIP_S, build_clips


def U(spk, s, e, text, **kw):
    return Utt(spk, text, text, s, e, **kw)


def test_clips_single_speaker_and_bounded():
    utts = []
    t = 0
    for i in range(60):  # one speaker rambling for ~4 minutes
        utts.append(U("A", t, t + 3500, "hola que tal " + "palabra " * 4))
        t += 4000
    clips = build_clips(utts, random.Random(0))
    assert clips
    for c in clips:
        assert c["duration"] <= MAX_CLIP_S + 0.5
        assert c["speaker"] == "A"


def test_no_cross_speaker_merge_and_overlap_drop():
    utts = [U("A", 0, 3000, "hello there my friend how are you"),
            U("B", 3100, 6000, "estoy muy bien gracias por preguntar"),
            U("A", 6100, 9000, "great to hear that news today")]
    clips = build_clips(utts, random.Random(1))
    assert {c["speaker"] for c in clips} == {"A", "B"}
    assert all(len(c["text"].split()) <= 8 for c in clips)  # never fused across speakers
    # heavy overlap with another speaker -> dropped
    utts = [U("A", 0, 5000, "one two three four five six seven"), U("B", 500, 4800, "quiero decir algo importante hoy")]
    clips = build_clips(utts, random.Random(1), max_overlap=0.15)
    assert not [c for c in clips if c["speaker"] == "A"]


def test_unintelligible_dropped():
    utts = [U("A", 0, 3000, "hello there my friend how", has_unintelligible=True)]
    assert build_clips(utts, random.Random(0)) == []
