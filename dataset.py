import json
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset


def load_tang_poems(filepath: str = "data/poet.tang.0.json") -> List[str]:
    """
    从指定 JSON 文件加载唐诗数据，提取每首诗的 paragraphs 字段合并为字符串，返回诗歌字符串列表。
    """
    poems = []
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            content = "".join(item.get("paragraphs", []))
            poems.append(content)
    return poems


def build_vocab(poems: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    构建字符词表，返回 char2idx 和 idx2char 字典。包含 <PAD>。
    """
    PAD_TOKEN = "<PAD>"
    chars = set()
    for poem in poems:
        chars.update(poem)
    chars = sorted(list(chars))
    chars = [PAD_TOKEN] + chars
    char2idx = {ch: idx for idx, ch in enumerate(chars)}
    idx2char = {idx: ch for idx, ch in enumerate(chars)}
    return char2idx, idx2char


def poems_to_indices(poems: List[str], char2idx: Dict[str, int]) -> List[List[int]]:
    """
    将诗歌列表转为索引编码。
    """
    return [[char2idx[ch] for ch in poem] for poem in poems]


class PoetryDataset(Dataset):
    def __init__(self, poems_indices: List[List[int]], seq_len: int):
        """
        poems_indices: 诗歌的索引编码列表
        seq_len: 每个样本的输入序列长度
        """
        self.seq_len = seq_len
        self.samples = []  # (input_seq, target_seq)
        for poem in poems_indices:
            if len(poem) < seq_len + 1:
                continue
            for i in range(len(poem) - seq_len):
                input_seq = poem[i:i+seq_len]
                target_seq = poem[i+1:i+seq_len+1]
                self.samples.append((input_seq, target_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, target_seq = self.samples[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)
