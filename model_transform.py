import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class PoetryTransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6,
                 dim_feedforward=512, dropout=0.1, pad_idx=0, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.pad_idx = pad_idx

    def forward(self, src, src_key_padding_mask=None):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        logits = self.fc_out(out)
        return logits

    @torch.no_grad()
    def generate(self, input_seq, max_gen_len, char2idx, idx2char,
                 temperature=1.0, top_k=0, top_p=0.0, device=None):
        if device is None:
            device = next(self.parameters()).device
        if isinstance(input_seq, list):
            input_seq = torch.tensor(input_seq, dtype=torch.long, device=device)
        input_seq = input_seq.unsqueeze(0)  # shape: (1, seq_len)

        generated = []  # 只生成新词
        full_seq = input_seq.tolist()[0].copy()

        for _ in range(max_gen_len):
            inp = torch.tensor([full_seq[-self.pos_encoder.pe.size(1):]], dtype=torch.long, device=device)
            logits = self.forward(inp)
            next_token_logits = logits[0, -1, :] / temperature

            # 避免采样 <PAD>
            next_token_logits[self.pad_idx] = -float('inf')
            probs = F.softmax(next_token_logits, dim=-1)

            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(probs, top_k)
                probs = torch.zeros_like(probs).scatter_(0, top_k_indices, top_k_values)
                probs /= probs.sum()
            elif top_p > 0.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=0)
                cutoff = cumulative_probs > top_p
                if torch.any(cutoff):
                    last_index = torch.where(cutoff)[0][0] + 1
                    probs[sorted_indices[last_index:]] = 0
                    probs /= probs.sum()

            next_token = torch.multinomial(probs, num_samples=1).item()
            full_seq.append(next_token)
            generated.append(next_token)

        # 只保留在 idx2char 中的 token，防止KeyError
        return ''.join([idx2char[idx] for idx in generated if idx != self.pad_idx and idx in idx2char])
