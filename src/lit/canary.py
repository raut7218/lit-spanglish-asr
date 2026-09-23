"""Canary-1b-v2 in plain PyTorch - no NeMo (the competition runtime has torch + transformers<5 but no NeMo).

    encoder: transformers' ParakeetEncoder (FastConformer, the same network as NeMo's ConformerEncoder)
    decoder: 8 pre-LN Transformer layers + fixed interleaved sinusoids (NeMo `TransformerDecoderNM` layout)
    frontend: NeMo's log-mel (pre-emphasis, 512-pt STFT, per-feature normalisation) with NeMo's own filterbank

The same module is used for training (lit.train), evaluation and the offline runtime (lit.infer), so there is no
train/inference drift. Weights come from `scripts/convert_canary.py` (a .nemo -> model dir with config.json,
model.safetensors, tokenizer.model). Prompt (Canary-2 format) + our extra style token:

    <|startofcontext|><|startoftranscript|><|emo:undefined|><|src|><|tgt|><|pnc|><|noitn|><|notimestamp|><|nodiarize|>[<|verbatim|>]
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

SR = 16000
_SPECIAL = re.compile(r"^<\|.*\|>$|^<pad>$|^<unk>$|^<s>$|^</s>$")


# --------------------------------------------------------------------------------------- tokenizer
class Tokenizer:
    """Canary's unified SentencePiece model; task/language tokens are pieces of the same vocabulary."""

    def __init__(self, path, verbatim: str | None = None):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor(model_file=str(path))
        self.special = {i for i in range(self.sp.get_piece_size()) if _SPECIAL.match(self.sp.id_to_piece(i))}
        self.verbatim = verbatim
        self.eos = self.id("<|endoftext|>")
        self.pad = self.sp.pad_id() if self.sp.pad_id() >= 0 else self.eos

    def id(self, piece: str) -> int:
        i = self.sp.piece_to_id(piece)
        if i == self.sp.unk_id() and piece != "<unk>":
            raise KeyError(f"{piece} is not in the Canary vocabulary")
        return i

    def prompt(self, src: str = "es", tgt: str | None = None, pnc: bool = True) -> list[int]:
        pieces = ["<|startofcontext|>", "<|startoftranscript|>", "<|emo:undefined|>", f"<|{src}|>", f"<|{tgt or src}|>",
                  "<|pnc|>" if pnc else "<|nopnc|>", "<|noitn|>", "<|notimestamp|>", "<|nodiarize|>"]
        if self.verbatim:
            pieces.append(self.verbatim)
        return [self.id(p) for p in pieces]

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text.strip())

    def decode(self, ids) -> str:
        return self.sp.decode([int(i) for i in ids if int(i) not in self.special]).strip()


# --------------------------------------------------------------------------------------- decoder
def sinusoids(n: int, d: int) -> torch.Tensor:
    """NeMo FixedPositionalEncoding: interleaved sin/cos, divided by sqrt(d)."""
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe / math.sqrt(d)


class Attention(nn.Module):
    def __init__(self, h: int, heads: int):
        super().__init__()
        self.heads = heads
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = (nn.Linear(h, h) for _ in range(4))

    def split(self, x):
        b, t, h = x.shape
        return x.view(b, t, self.heads, h // self.heads).transpose(1, 2)

    def kv(self, x):
        return self.split(self.k_proj(x)), self.split(self.v_proj(x))

    def forward(self, x, k, v, mask=None, causal=False):
        o = F.scaled_dot_product_attention(self.split(self.q_proj(x)), k, v, attn_mask=mask, is_causal=causal)
        return self.o_proj(o.transpose(1, 2).reshape(x.shape))


class DecoderLayer(nn.Module):
    def __init__(self, h: int, heads: int, inner: int, dropout: float):
        super().__init__()
        self.input_layernorm, self.post_attention_layernorm, self.final_layernorm = (nn.LayerNorm(h) for _ in range(3))
        self.self_attn, self.encoder_attn = Attention(h, heads), Attention(h, heads)
        self.fc1, self.fc2 = nn.Linear(h, inner), nn.Linear(inner, h)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cross_kv, enc_mask, cache=None):
        """cache: None (training, full causal pass) or a dict holding this layer's self-attn k/v (grown in place)."""
        y = self.input_layernorm(x)
        k, v = self.self_attn.kv(y)
        if cache is None:
            x = x + self.drop(self.self_attn(y, k, v, causal=True))
        else:
            if "k" in cache:
                k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
            x = x + self.self_attn(y, k, v, causal=x.shape[1] > 1 and k.shape[2] == x.shape[1])
        x = x + self.drop(self.encoder_attn(self.post_attention_layernorm(x), *cross_kv, mask=enc_mask))
        return x + self.drop(self.fc2(self.drop(F.relu(self.fc1(self.final_layernorm(x))))))


# --------------------------------------------------------------------------------------- model
class CanaryModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        from transformers import ParakeetEncoder, ParakeetEncoderConfig

        self.cfg = cfg
        ec = ParakeetEncoderConfig(**cfg["encoder"])
        ec._attn_implementation = "sdpa"
        self.encoder = ParakeetEncoder(ec)
        d = cfg["decoder"]
        h, vocab = d["hidden_size"], d["vocab_size"]
        self.embed_tokens = nn.Embedding(vocab, h)
        self.register_buffer("pos", sinusoids(d["max_position_embeddings"], h), persistent=False)
        self.embedding_layernorm = nn.LayerNorm(h)
        self.layers = nn.ModuleList(DecoderLayer(h, d["num_attention_heads"], d["intermediate_size"], d.get("dropout", 0.1))
                                    for _ in range(d["num_hidden_layers"]))
        self.norm = nn.LayerNorm(h)
        self.proj_out = nn.Linear(h, vocab)
        if d.get("tie_word_embeddings", True):
            self.proj_out.weight = self.embed_tokens.weight
        p = cfg["preprocessor"]
        self.register_buffer("mel_fb", torch.zeros(p["features"], p["n_fft"] // 2 + 1))  # overwritten by the checkpoint
        self.register_buffer("window", torch.hann_window(p["win_length"], periodic=False), persistent=False)
        self.emb_drop = nn.Dropout(d.get("dropout", 0.1))

    # ---- frontend
    @torch.no_grad()
    def features(self, wav: torch.Tensor, lengths: torch.Tensor):
        """(B, N) float waveform at 16 kHz + sample lengths -> (B, T, n_mels) normalised log-mel, (B, T) bool mask."""
        p = self.cfg["preprocessor"]
        with torch.autocast(device_type=wav.device.type, enabled=False):
            x = wav.float()
            n = torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]
            x = torch.cat([x[:, :1], x[:, 1:] - p["preemph"] * x[:, :-1]], 1).masked_fill(~n, 0.0)
            spec = torch.stft(x, p["n_fft"], hop_length=p["hop_length"], win_length=p["win_length"], window=self.window,
                              center=True, pad_mode="constant", return_complex=True)
            mel = torch.log(self.mel_fb @ (spec.real ** 2 + spec.imag ** 2) + 2.0 ** -24).transpose(1, 2)  # (B,T,M)
            flen = torch.div(lengths + p["n_fft"] // 2 * 2 - p["n_fft"], p["hop_length"], rounding_mode="floor").clamp(min=2)
            mask = torch.arange(mel.shape[1], device=x.device)[None, :] < flen[:, None]
            m = mask[..., None]
            mean = (mel * m).sum(1, keepdim=True) / flen[:, None, None]
            var = (((mel - mean) * m) ** 2).sum(1, keepdim=True) / (flen - 1)[:, None, None]
            return ((mel - mean) / (var.sqrt() + 1e-5)).masked_fill(~m, 0.0), mask

    # ---- encoder / decoder
    def encode(self, feats: torch.Tensor, mask: torch.Tensor):
        enc = self.encoder(input_features=feats, attention_mask=mask.long()).last_hidden_state
        enc_mask = self.encoder._get_output_attention_mask(mask.long(), target_length=enc.shape[1])
        return enc, enc_mask

    def cross_kv(self, enc):
        return [layer.encoder_attn.kv(enc) for layer in self.layers]

    def decode(self, ids, cross, enc_mask, caches=None, start: int = 0):
        x = self.embedding_layernorm(self.embed_tokens(ids) + self.pos[start : start + ids.shape[1]].to(self.embed_tokens.weight.dtype))
        x = self.emb_drop(x)
        am = enc_mask[:, None, None, :]
        for i, layer in enumerate(self.layers):
            x = layer(x, cross[i], am, None if caches is None else caches[i])
        return self.proj_out(self.norm(x))

    def forward(self, feats, mask, dec_in, labels, label_smoothing: float = 0.0):
        enc, enc_mask = self.encode(feats, mask)
        logits = self.decode(dec_in, self.cross_kv(enc), enc_mask)
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100,
                               label_smoothing=label_smoothing)

    def train(self, mode: bool = True):
        # BatchNorm stays frozen: small padded fine-tuning batches would corrupt the pretrained running statistics
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()
        return self

    def freeze_encoder_below(self, k: int):
        """No gradients for the conv subsampling + encoder layers < k (they then store no activations either)."""
        for p in self.encoder.subsampling.parameters():
            p.requires_grad_(k <= 0)
        for i, layer in enumerate(self.encoder.layers):
            for p in layer.parameters():
                p.requires_grad_(i >= k)

    # ---- decoding
    @torch.no_grad()
    def generate(self, feats, mask, prompts: list[list[int]], eos: int, beam: int = 1, max_new: int = 200,
                 len_norm: float = 1.0):
        """Batched beam search (beam=1 is greedy). prompts: one prompt per clip, all the same length.
        Returns [(token_ids, score)] per clip; score = sum log-prob / len**len_norm."""
        enc, enc_mask = self.encode(feats, mask)
        B, K = enc.shape[0], beam
        enc, enc_mask = enc.repeat_interleave(K, 0), enc_mask.repeat_interleave(K, 0)
        cross = self.cross_kv(enc)
        caches = [{} for _ in self.layers]
        ids = torch.tensor(prompts, device=enc.device).repeat_interleave(K, 0)
        logits = self.decode(ids, cross, enc_mask, caches, 0)[:, -1]
        pos = ids.shape[1]
        scores = torch.zeros(B, K, device=enc.device)
        scores[:, 1:] = -float("inf")  # all beams start identical: expand only the first
        seqs = torch.zeros(B * K, 0, dtype=torch.long, device=enc.device)
        done = [[] for _ in range(B)]
        finished = [False] * B
        for step in range(max_new):
            logp = F.log_softmax(logits.float(), -1)
            V = logp.shape[-1]
            top_s, top_i = (scores.view(-1, 1) + logp).view(B, K * V).topk(2 * K, -1)
            new_scores = torch.full((B, K), -float("inf"), device=enc.device)
            src = torch.arange(B * K, device=enc.device).view(B, K)  # dead rows just keep their own cache
            tok = torch.full((B, K), eos, dtype=torch.long, device=enc.device)
            ts, ti = top_s.tolist(), top_i.tolist()
            for b in range(B):
                if finished[b]:
                    continue
                j = 0
                for s, i in zip(ts[b], ti[b]):
                    if s == -float("inf") or j == K:
                        break
                    k, t = divmod(i, V)
                    if t == eos:  # a finished hypothesis (length = generated tokens incl. eos)
                        done[b].append((seqs[b * K + k].tolist(), s / (step + 1) ** len_norm))
                    else:
                        new_scores[b, j], src[b, j], tok[b, j] = s, b * K + k, t
                        j += 1
                finished[b] = len(done[b]) >= K or j == 0
            if all(finished):
                break
            flat = src.view(-1)
            seqs = torch.cat([seqs[flat], tok.view(-1, 1)], 1)
            for c in caches:
                c["k"], c["v"] = c["k"][flat], c["v"][flat]
            scores = new_scores
            logits = self.decode(tok.view(-1, 1), cross, enc_mask, caches, pos)[:, -1]
            pos += 1
        out = []
        for b in range(B):
            hyps = done[b]
            if not hyps:  # hit max_new without an eos: take the best live beam
                k = int(scores[b].argmax())
                hyps = [(seqs[b * K + k].tolist(), scores[b, k].item() / max(1, seqs.shape[1]) ** len_norm)]
            out.append(max(hyps, key=lambda h: h[1]))
        return out


# --------------------------------------------------------------------------------------- io
def load(model_dir, device="cpu", dtype=torch.float32):
    from safetensors.torch import load_file

    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    model = CanaryModel(cfg)
    sd = load_file(str(model_dir / "model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k != "proj_out.weight" or "proj_out.weight" in sd]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
    model.to(device=device, dtype=dtype)
    model.mel_fb.data = model.mel_fb.data.float()  # the frontend always runs in fp32
    model.window.data = model.window.data.float()
    tok = Tokenizer(model_dir / "tokenizer.model", cfg.get("verbatim_token"))
    return model.eval(), tok


def save(model: CanaryModel, model_dir, tokenizer_path=None, dtype=None):
    from safetensors.torch import save_file

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().to("cpu", dtype=dtype if dtype and v.is_floating_point() and k != "mel_fb" else v.dtype).contiguous()
          for k, v in model.state_dict().items()}
    if model.proj_out.weight is model.embed_tokens.weight:
        sd.pop("proj_out.weight")  # tied; safetensors refuses shared tensors
    save_file(sd, str(model_dir / "model.safetensors"))
    (model_dir / "config.json").write_text(json.dumps(model.cfg, indent=2))
    if tokenizer_path and Path(tokenizer_path).resolve() != (model_dir / "tokenizer.model").resolve():
        import shutil

        shutil.copy(tokenizer_path, model_dir / "tokenizer.model")


def pick_dtype(name: str = "auto"):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    return dict(fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32)[name]


def pad_waves(arrs: list[np.ndarray]):
    n = max(len(a) for a in arrs)
    out = np.zeros((len(arrs), n), np.float32)
    for i, a in enumerate(arrs):
        out[i, : len(a)] = a
    return torch.from_numpy(out), torch.tensor([len(a) for a in arrs], dtype=torch.long)


@torch.no_grad()
def transcribe(model: CanaryModel, tok: Tokenizer, audios: list[np.ndarray], languages=("es",), beam: int = 1,
               batch_size: int = 8, max_s: float = 40.0, tokens_per_s: float = 8.0, amp_dtype=None):
    """Transcribe arrays; clips > max_s are split at quiet points and re-joined. With several languages every clip
    is decoded once per language prompt and the hypothesis with the best normalised log-prob wins."""
    from .audio import split_long

    device = next(model.parameters()).device
    pieces, owner = [], []
    for i, a in enumerate(audios):
        for c in split_long(a, max_s=max_s):
            pieces.append(c)
            owner.append(i)
    best = [("", -float("inf"))] * len(pieces)
    order = np.argsort([-len(p) for p in pieces])
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        wav, lens = pad_waves([pieces[i] for i in idx])
        feats, mask = model.features(wav.to(device), lens.to(device))
        max_new = int(min(440, tokens_per_s * lens.max().item() / SR + 24))
        for lang in languages:
            prompt = tok.prompt(lang)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                res = model.generate(feats, mask, [prompt] * len(idx), tok.eos, beam=beam, max_new=max_new)
            for i, (ids, score) in zip(idx, res):
                if score > best[i][1]:
                    best[i] = (tok.decode(ids), score)
    merged = [""] * len(audios)
    for o, (text, _) in zip(owner, best):
        merged[o] = (merged[o] + " " + text).strip()
    return merged
