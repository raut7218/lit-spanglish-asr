"""Tiny random-weight Canary: frontend == transformers' ParakeetFeatureExtractor, padding invariance, KV cache ==
full pass, beam search sanity, save/load round trip, tokenizer prompt. (NeMo parity: test_canary_parity.py.)"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
spm = pytest.importorskip("sentencepiece")

from lit import canary  # noqa: E402

N_MELS = 16
SPECIALS = ["<|endoftext|>", "<|startofcontext|>", "<|startoftranscript|>", "<|emo:undefined|>", "<|es|>", "<|en|>",
            "<|pnc|>", "<|nopnc|>", "<|noitn|>", "<|notimestamp|>", "<|nodiarize|>", "<|spltoken0|>"]


def tiny_cfg(vocab):
    return dict(
        encoder=dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, num_mel_bins=N_MELS,
                     subsampling_conv_channels=8, conv_kernel_size=9, scale_input=False, layerdrop=0.0, dropout=0.0,
                     attention_dropout=0.0, activation_dropout=0.0),
        decoder=dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, vocab_size=vocab,
                     max_position_embeddings=256, dropout=0.0),
        preprocessor=dict(features=N_MELS, n_fft=512, win_length=400, hop_length=160, preemph=0.97),
    )


@pytest.fixture(scope="module")
def model_tok(spm_path):
    torch.manual_seed(0)
    tok = canary.Tokenizer(spm_path, "<|spltoken0|>")
    m = canary.CanaryModel(tiny_cfg(tok.sp.get_piece_size()))
    from transformers import ParakeetFeatureExtractor

    fe = ParakeetFeatureExtractor(feature_size=N_MELS)
    m.mel_fb.copy_(fe.mel_filters)
    for mod in m.modules():  # non-trivial BN statistics so masking bugs show up
        if isinstance(mod, torch.nn.BatchNorm1d):
            mod.running_mean.normal_(0, 0.1); mod.running_var.uniform_(0.5, 1.5)
    return m.eval(), tok


def _wave(sec, seed):
    return (np.random.default_rng(seed).standard_normal(int(sec * 16000)) * 0.1).astype(np.float32)


def test_features_match_parakeet_extractor(model_tok):
    from transformers import ParakeetFeatureExtractor

    m, _ = model_tok
    waves = [_wave(3.0, 1), _wave(1.7, 2)]
    ref = ParakeetFeatureExtractor(feature_size=N_MELS)(waves, sampling_rate=16000, return_tensors="pt")
    wav, lens = canary.pad_waves(waves)
    got, mask = m.features(wav, lens)
    T = ref["input_features"].shape[1]
    assert torch.equal(mask[:, :T], ref["attention_mask"].bool())
    assert (got[:, :T] - ref["input_features"]).abs().max() < 1e-3


def test_padding_does_not_change_outputs(model_tok):
    m, tok = model_tok
    short, long_ = _wave(1.5, 3), _wave(4.0, 4)
    f1, m1 = m.features(*canary.pad_waves([short]))
    fb, mb = m.features(*canary.pad_waves([short, long_]))
    e1, _ = m.encode(f1, m1)
    eb, emb = m.encode(fb, mb)
    n = e1.shape[1]
    assert torch.isfinite(eb[0, :n]).all()
    assert (eb[0, :n] - e1[0]).abs().max() < 1e-4
    assert int(emb[0].sum()) == n


def test_kv_cache_matches_full_pass(model_tok):
    m, tok = model_tok
    f, mk = m.features(*canary.pad_waves([_wave(2.0, 5)]))
    enc, em = m.encode(f, mk)
    cross = m.cross_kv(enc)
    ids = torch.tensor([tok.prompt("es") + tok.encode("hola que tal")])
    full = m.decode(ids, cross, em)
    caches = [{} for _ in m.layers]
    P = 5
    parts = [m.decode(ids[:, :P], cross, em, caches, 0)]
    for t in range(P, ids.shape[1]):
        parts.append(m.decode(ids[:, t : t + 1], cross, em, caches, t))
    assert (torch.cat(parts, 1) - full).abs().max() < 1e-4


def test_generate_greedy_matches_argmax_and_beam_runs(model_tok):
    m, tok = model_tok
    wav, lens = canary.pad_waves([_wave(2.0, 6), _wave(1.2, 7)])
    f, mk = m.features(wav, lens)
    prompt = tok.prompt("en")
    (g0, _), (g1, _) = m.generate(f, mk, [prompt, prompt], tok.eos, beam=1, max_new=12)
    # reference greedy loop without cache, clip 0 alone
    f0, m0 = m.features(*canary.pad_waves([_wave(2.0, 6)]))
    enc, em = m.encode(f0, m0)
    cross, ids = m.cross_kv(enc), list(prompt)
    for _ in range(12):
        t = int(m.decode(torch.tensor([ids]), cross, em)[0, -1].argmax())
        if t == tok.eos:
            break
        ids.append(t)
    assert g0 == ids[len(prompt):]
    beams = m.generate(f, mk, [prompt, prompt], tok.eos, beam=3, max_new=12)
    assert len(beams) == 2 and all(isinstance(s, float) for _, s in beams)


def test_training_loss_backward_and_freeze(model_tok):
    m, tok = model_tok
    m.freeze_encoder_below(1)
    m.train()
    assert not any(mod.training for mod in m.modules() if isinstance(mod, torch.nn.BatchNorm1d))
    f, mk = m.features(*canary.pad_waves([_wave(2.0, 8), _wave(1.0, 9)]))
    seq = tok.prompt("es") + tok.encode("hola que tal") + [tok.eos]
    dec_in, labels = torch.tensor([seq[:-1]] * 2), torch.tensor([seq[1:]] * 2)
    loss = m(f, mk, dec_in, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert m.encoder.layers[0].feed_forward1.linear1.weight.grad is None
    assert m.encoder.layers[1].feed_forward1.linear1.weight.grad is not None
    m.zero_grad(set_to_none=True); m.freeze_encoder_below(0); m.eval()


def test_save_load_roundtrip(model_tok, spm_path, tmp_path):
    m, tok = model_tok
    canary.save(m, tmp_path, tokenizer_path=spm_path)
    import json

    cfg = json.loads((tmp_path / "config.json").read_text())
    cfg["verbatim_token"] = "<|spltoken0|>"
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    m2, tok2 = canary.load(tmp_path)
    assert m2.proj_out.weight is m2.embed_tokens.weight
    for k, v in m.state_dict().items():
        assert torch.equal(v, m2.state_dict()[k]), k
    assert tok2.prompt("es")[-1] == tok2.id("<|spltoken0|>")


def test_tokenizer_prompt_and_decode(spm_path):
    tok = canary.Tokenizer(spm_path)
    p = tok.prompt("es", pnc=True)
    assert [tok.sp.id_to_piece(i) for i in p][:6] == ["▁", "<|startofcontext|>", "<|startoftranscript|>", "<|emo:undefined|>", "<|es|>", "<|es|>"]
    assert tok.decode(p + tok.encode("hola que tal") + [tok.eos]) == "hola que tal"
    with pytest.raises(KeyError):
        tok.id("<|xx|>")


def _to_nemo_key(k):
    import re

    enc = [(r"^encoder\.subsampling\.layers\.", "encoder.pre_encode.conv."), (r"^encoder\.subsampling\.linear\.", "encoder.pre_encode.out."),
           (r"\.conv\.norm\.", ".conv.batch_norm."), (r"\.q_proj\.", ".linear_q."), (r"\.k_proj\.", ".linear_k."),
           (r"\.v_proj\.", ".linear_v."), (r"\.o_proj\.", ".linear_out."), (r"\.relative_k_proj\.", ".linear_pos."),
           (r"\.bias_([uv])$", r".pos_bias_\1")]
    dec = [(r"^embed_tokens\.", "transf_decoder._embedding.token_embedding."),
           (r"^embedding_layernorm\.", "transf_decoder._embedding.layer_norm."), (r"^norm\.", "transf_decoder._decoder.final_layer_norm."),
           (r"^proj_out\.", "log_softmax.mlp.layer0."), (r"^layers\.(\d+)\.", r"transf_decoder._decoder.layers.\1."),
           (r"\.input_layernorm\.", ".layer_norm_1."), (r"\.post_attention_layernorm\.", ".layer_norm_2."),
           (r"\.final_layernorm\.", ".layer_norm_3."), (r"\.self_attn\.", ".first_sub_layer."), (r"\.encoder_attn\.", ".second_sub_layer."),
           (r"\.fc1\.", ".third_sub_layer.dense_in."), (r"\.fc2\.", ".third_sub_layer.dense_out."),
           (r"\.q_proj\.", ".query_net."), (r"\.k_proj\.", ".key_net."), (r"\.v_proj\.", ".value_net."), (r"\.o_proj\.", ".out_projection.")]
    for p, r in (enc if k.startswith("encoder.") else dec):
        k = re.sub(p, r, k)
    return k


def test_convert_from_fake_nemo(model_tok, spm_path, tmp_path):
    import sys
    import tarfile
    from pathlib import Path

    import yaml

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import convert_canary

    m, tok = model_tok
    c = tiny_cfg(tok.sp.get_piece_size())
    sd = {_to_nemo_key(k): v for k, v in m.state_dict().items() if k not in ("mel_fb",)}
    sd["preprocessor.featurizer.fb"] = m.mel_fb[None].clone()
    sd["preprocessor.featurizer.window"] = torch.hann_window(400, periodic=False)
    sd["transf_decoder._embedding.position_embedding.pos_enc"] = canary.sinusoids(256, 32)
    sd["encoder.pos_enc.pe"] = torch.zeros(1, 9, 32)
    d = c["decoder"]
    nc = dict(
        encoder=dict(d_model=32, n_layers=2, n_heads=2, ff_expansion_factor=2, feat_in=N_MELS, conv_kernel_size=9, subsampling_factor=8,
                     subsampling_conv_channels=8, xscaling=False, subsampling="dw_striding", self_attention_model="rel_pos"),
        preprocessor=dict(sample_rate=16000, features=N_MELS, n_fft=512, window_size=0.025, window_stride=0.01, normalize="per_feature"),
        transf_decoder=dict(config_dict=dict(hidden_size=32, num_layers=2, num_attention_heads=2, inner_size=64,
                                             max_sequence_length=256, pre_ln=True, hidden_act="relu")),
        head=dict(num_classes=d["vocab_size"]), tokenizer=dict(model_path="nemo:abc_tokenizer.model"))
    src = tmp_path / "src"
    src.mkdir()
    (src / "model_config.yaml").write_text(yaml.safe_dump(nc))
    torch.save(sd, src / "model_weights.ckpt")
    (src / "abc_tokenizer.model").write_bytes(Path(spm_path).read_bytes())
    nemo = tmp_path / "fake.nemo"
    with tarfile.open(nemo, "w") as t:
        for f in src.iterdir():
            t.add(f, arcname=f.name)
    out = tmp_path / "conv"
    convert_canary.convert(nemo, out, "auto")
    m2, tok2 = canary.load(out)
    assert tok2.verbatim == "<|spltoken0|>"
    v, pnc = tok2.id("<|spltoken0|>"), tok2.id("<|pnc|>")
    assert torch.equal(m2.embed_tokens.weight[v], m2.embed_tokens.weight[pnc])
    ref = m.state_dict()
    for k, val in m2.state_dict().items():
        if k in ("embed_tokens.weight", "proj_out.weight"):
            continue
        assert torch.equal(val, ref[k]), k
