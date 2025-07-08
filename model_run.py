import torch
import torch.nn as nn
import torch.nn.functional as F

class PoetryRNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2, dropout=0.5, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pad_idx = pad_idx

    def forward(self, x, hidden=None):
        embed = self.embedding(x)
        output, hidden = self.lstm(embed, hidden)
        logits = self.fc(output)
        return logits, hidden

    @torch.no_grad()
    def generate(self, input_seq, max_gen_len, char2idx, idx2char,
                 temperature=1.0, top_k=0, top_p=0.0, device=None):
        """
        生成新内容，只输出词表里的字，未登录字用“□”占位，不会KeyError。
        """
        if device is None:
            device = next(self.parameters()).device
        if isinstance(input_seq, list):
            input_seq = torch.tensor(input_seq, dtype=torch.long, device=device)
        input_seq = input_seq.unsqueeze(0)  # shape: (1, seq_len)

        full_seq = input_seq.tolist()[0].copy()
        generated = []

        hidden = None

        for _ in range(max_gen_len):
            logits, hidden = self.forward(input_seq, hidden)
            logits = logits[:, -1, :] / temperature
            logits[0, self.pad_idx] = -float('inf')
            probs = F.softmax(logits, dim=-1).squeeze(0)

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
            input_seq = torch.tensor([[next_token]], dtype=torch.long, device=device)

        # ★★★ 这里加上 KeyError 容错，只显示词表里有的字，没有的用 '□'
        return ''.join([idx2char[idx] if idx in idx2char else '□' for idx in generated if idx != self.pad_idx])
