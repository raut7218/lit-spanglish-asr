"""Convert nvidia/canary-1b-v2 (.nemo) into a NeMo-free model dir for lit.canary, and optionally dump NeMo golden
outputs for the parity gate (tests/test_canary_parity.py).

    python scripts/convert_canary.py --out /content/canary_base                     # no NeMo needed
    python scripts/convert_canary.py --out /content/canary_base --golden a.flac b.flac  # needs nemo_toolkit[asr]

The .nemo archive is a tar of model_config.yaml + model_weights.ckpt + the SentencePiece model. Keys are renamed onto
transformers' ParakeetEncoder and our decoder (lit/canary.py); every checkpoint key must be consumed and every model
key filled, and the fixed tables we recompute (decoder sinusoids, STFT window) are checked against the checkpoint.
"""

import argparse
import re
import sys
import tarfile
import tempfile
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lit import canary  # noqa: E402

ENC_MAP = [
    (r"^encoder\.pre_encode\.conv\.", "encoder.subsampling.layers."),
    (r"^encoder\.pre_encode\.out\.", "encoder.subsampling.linear."),
    (r"^encoder\.layers\.(\d+)\.conv\.batch_norm\.", r"encoder.layers.\1.conv.norm."),
    (r"linear_([kv])\.", r"\1_proj."),
    (r"linear_out\.", "o_proj."),
    (r"linear_q\.", "q_proj."),
    (r"pos_bias_([uv])$", r"bias_\1"),
    (r"linear_pos\.", "relative_k_proj."),
]
DEC_MAP = [
    (r"^transf_decoder\._embedding\.token_embedding\.", "embed_tokens."),
    (r"^transf_decoder\._embedding\.layer_norm\.", "embedding_layernorm."),
    (r"^transf_decoder\._decoder\.final_layer_norm\.", "norm."),
    (r"^transf_decoder\._decoder\.layers\.(\d+)\.layer_norm_1\.", r"layers.\1.input_layernorm."),
    (r"^transf_decoder\._decoder\.layers\.(\d+)\.layer_norm_2\.", r"layers.\1.post_attention_layernorm."),
    (r"^transf_decoder\._decoder\.layers\.(\d+)\.layer_norm_3\.", r"layers.\1.final_layernorm."),
    (r"^transf_decoder\._decoder\.layers\.(\d+)\.first_sub_layer\.", r"layers.\1.self_attn."),
    (r"^transf_decoder\._decoder\.layers\.(\d+)\.second_sub_layer\.", r"layers.\1.encoder_attn."),
    (r"^transf_decoder\._decoder\.layers\.(\d+)\.third_sub_layer\.dense_in\.", r"layers.\1.fc1."),
    (r"^transf_decoder\._decoder\.layers\.(\d+)\.third_sub_layer\.dense_out\.", r"layers.\1.fc2."),
    (r"query_net\.", "q_proj."), (r"key_net\.", "k_proj."), (r"value_net\.", "v_proj."), (r"out_projection\.", "o_proj."),
    (r"^log_softmax\.mlp\.layer0\.", "proj_out."),
]


def rename(key, rules):
    for pat, rep in rules:
        key = re.sub(pat, rep, key)
    return key


def build_cfg(nc: dict) -> dict:
    e, p = nc["encoder"], nc["preprocessor"]
    guard = p.get("log_zero_guard_value", 2 ** -24)
    if str(guard).replace(" ", "") == "2**-24":  # NeMo yaml often keeps the expression as a string
        guard = 2 ** -24
    checks = {"subsampling": (e.get("subsampling"), "dw_striding"), "self_attention_model": (e.get("self_attention_model"), "rel_pos"),
              "conv_norm_type": (e.get("conv_norm_type", "batch_norm"), "batch_norm"), "normalize": (p.get("normalize"), "per_feature"),
              "window": (p.get("window", "hann"), "hann"), "mag_power": (float(p.get("mag_power", 2.0)), 2.0),
              "log_zero_guard_value": (float(guard), 2 ** -24),
              "exact_pad": (bool(p.get("exact_pad", False)), False)}
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    assert not bad, f"unsupported NeMo config (port assumes the right-hand values): {bad}"
    sr = p["sample_rate"]
    d = nc["transf_decoder"]["config_dict"]
    assert d.get("pre_ln", True) and d.get("hidden_act", "relu") == "relu", "decoder must be pre-LN / relu"
    return dict(
        encoder=dict(hidden_size=e["d_model"], num_hidden_layers=e["n_layers"], num_attention_heads=e["n_heads"],
                     intermediate_size=e["d_model"] * e.get("ff_expansion_factor", 4), num_mel_bins=e["feat_in"],
                     conv_kernel_size=e["conv_kernel_size"], subsampling_factor=e["subsampling_factor"],
                     subsampling_conv_channels=e["subsampling_conv_channels"], max_position_embeddings=e.get("pos_emb_max_len", 5000),
                     scale_input=bool(e.get("xscaling", True)), attention_bias=bool(e.get("use_bias", True)), layerdrop=0.0,
                     dropout=e.get("dropout", 0.1), dropout_positions=e.get("dropout_emb", 0.0), attention_dropout=e.get("dropout_att", 0.1),
                     activation_dropout=e.get("dropout", 0.1)),
        decoder=dict(hidden_size=d["hidden_size"], num_hidden_layers=d["num_layers"], num_attention_heads=d["num_attention_heads"],
                     intermediate_size=d["inner_size"], vocab_size=nc["head"]["num_classes"],
                     max_position_embeddings=d["max_sequence_length"], dropout=d.get("ffn_dropout", 0.1)),
        preprocessor=dict(features=p["features"], n_fft=p["n_fft"], win_length=int(p["window_size"] * sr),
                          hop_length=int(p["window_stride"] * sr), preemph=float(p.get("preemph", 0.97)), sample_rate=sr),
    )


def convert(nemo_path: Path, out: Path, verbatim: str | None):
    tmp = Path(tempfile.mkdtemp())
    with tarfile.open(nemo_path) as t:
        t.extractall(tmp)
    nc = yaml.safe_load((tmp / "model_config.yaml").read_text())
    cfg = build_cfg(nc)
    sd = torch.load(tmp / "model_weights.ckpt", map_location="cpu", weights_only=True)
    tok_path = tmp / nc["tokenizer"]["model_path"].split("nemo:")[-1]
    assert tok_path.exists(), f"tokenizer {tok_path} missing (aggregate tokenizers are not supported)"

    out_sd = {}
    for k, v in sd.items():
        if k.startswith("encoder."):
            if k.endswith("pos_enc.pe"):
                # recomputed by ParakeetEncoderRelPositionalEncoding
                continue
            out_sd[rename(k, ENC_MAP)] = v
        elif k.startswith(("transf_decoder.", "log_softmax.")):
            if k.endswith("position_embedding.pos_enc"):
                ours = canary.sinusoids(v.shape[0], v.shape[1])
                assert torch.allclose(v.float(), ours, atol=1e-5), "decoder sinusoid formula differs from NeMo"
                continue
            out_sd[rename(k, DEC_MAP)] = v
        elif k == "preprocessor.featurizer.fb":
            out_sd["mel_fb"] = v.reshape(v.shape[-2], v.shape[-1]).float()
        elif k == "preprocessor.featurizer.window":
            assert torch.allclose(v.float(), torch.hann_window(cfg["preprocessor"]["win_length"], periodic=False), atol=1e-6)
        else:
            raise ValueError(f"unhandled NeMo key {k}")
    tied = torch.equal(out_sd["proj_out.weight"], out_sd["embed_tokens.weight"])
    cfg["decoder"]["tie_word_embeddings"] = tied
    if tied:
        out_sd.pop("proj_out.weight")

    tok = canary.Tokenizer(tok_path)
    if verbatim == "auto":
        spare = [tok.sp.id_to_piece(i) for i in sorted(tok.special) if re.match(r"^<\|spltoken\d+\|>$", tok.sp.id_to_piece(i))]
        assert spare, "no <|spltokenN|> slot in the vocabulary: pass --verbatim_token none or an existing unused piece"
        verbatim = spare[0]
    cfg["verbatim_token"] = None if verbatim in (None, "none") else verbatim

    model = canary.CanaryModel(cfg)
    missing, unexpected = model.load_state_dict(out_sd, strict=False)
    missing = [k for k in missing if not (tied and k == "proj_out.weight")]
    assert not missing and not unexpected, f"missing={missing[:10]} unexpected={unexpected[:10]}"
    if cfg["verbatim_token"]:  # start the new style token as a copy of <|pnc|> (a neutral, never-predicted task token)
        with torch.no_grad():
            model.embed_tokens.weight[tok.id(cfg["verbatim_token"])] = model.embed_tokens.weight[tok.id("<|pnc|>")]
    canary.save(model, out, tokenizer_path=tok_path)
    (out / "nemo_model_config.yaml").write_text((tmp / "model_config.yaml").read_text())
    n = sum(p.numel() for p in model.parameters())
    print(f"[convert] {n/1e6:.0f}M params, tied={tied}, verbatim={cfg['verbatim_token']}, special tokens={len(tok.special)} -> {out}")
    return tmp


def golden(nemo_path: Path, out: Path, clips: list[str]):
    """NeMo reference: features, encoder states and greedy es/en transcripts for a few clips (fp32, CPU/GPU)."""
    from nemo.collections.asr.models import ASRModel

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from lit.audio import load_audio

    torch.backends.cuda.matmul.allow_tf32 = False  # true fp32 reference (cuDNN convs default to TF32 otherwise)
    torch.backends.cudnn.allow_tf32 = False
    m = ASRModel.restore_from(str(nemo_path), map_location="cuda" if torch.cuda.is_available() else "cpu").eval().float()
    dev = next(m.parameters()).device
    rec = dict(clips=[str(c) for c in clips], features=[], encoder=[], text={})
    with torch.no_grad():
        for c in clips:
            x = torch.from_numpy(load_audio(c))[None].to(dev)
            f, fl = m.preprocessor(input_signal=x, length=torch.tensor([x.shape[1]], device=dev))
            e, el = m.encoder(audio_signal=f, length=fl)
            rec["features"].append(f[0, :, : int(fl)].T.cpu())
            rec["encoder"].append(e[0, :, : int(el)].T.cpu())
    try:
        dc = m.cfg.decoding
        dc.beam.beam_size = 1
        m.change_decoding_strategy(dc)
    except Exception as ex:  # older/newer NeMo: default decoding is used, noted in the record
        rec["decoding_note"] = repr(ex)
    for lang in ("es", "en"):
        hyps = m.transcribe([str(c) for c in clips], source_lang=lang, target_lang=lang, batch_size=1)  # pnc is on by default
        rec["text"][lang] = [h.text if hasattr(h, "text") else str(h) for h in hyps]
    torch.save(rec, out / "golden.pt")
    print(f"[golden] {len(clips)} clips -> {out/'golden.pt'}; es: {rec['text']['es'][:2]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", help="path to canary-1b-v2.nemo (default: download from the HF hub)")
    ap.add_argument("--repo", default="nvidia/canary-1b-v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--verbatim_token", default="auto", help="'auto' = first <|spltokenN|>; 'none' = no style token")
    ap.add_argument("--golden", nargs="*", default=[], help="audio files for the NeMo parity reference (needs NeMo)")
    a = ap.parse_args()
    nemo = Path(a.nemo) if a.nemo else None
    if nemo is None:
        from huggingface_hub import hf_hub_download

        nemo = Path(hf_hub_download(a.repo, f"{a.repo.split('/')[-1]}.nemo"))
    out = Path(a.out)
    if (out / "model.safetensors").exists():  # --golden runs in NeMo's own venv: never re-convert there
        print(f"[convert] {out} already converted; delete it to convert again")
    else:
        convert(nemo, out, a.verbatim_token)
    if a.golden:
        golden(nemo, out, a.golden)


if __name__ == "__main__":
    main()
