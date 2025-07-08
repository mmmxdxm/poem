import argparse
import torch
from preprocess.dataset import load_tang_poems, build_vocab
from models.model_run import PoetryRNNModel
from models.model_transform import PoetryTransformerModel

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Chinese Poetry with Trained Model")
    parser.add_argument('--model', type=str, default='transformer', choices=['rnn', 'transformer'], help='Model type')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt string, e.g. 白日依山尽')
    parser.add_argument('--length', type=int, default=32, help='Length of generated poem')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--checkpoint', type=str, default=None, help='Model checkpoint path')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu)')
    return parser.parse_args()

def main():
    args = parse_args()
    # 设备
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    # 词表
    poems = load_tang_poems()
    char2idx, idx2char = build_vocab(poems)
    pad_idx = char2idx['<PAD>']
    vocab_size = len(char2idx)
    # 检查点路径
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = f'checkpoints/{args.model}_epoch10.pt'
    # 初始化模型
    if args.model == 'rnn':
        model = PoetryRNNModel(
            vocab_size=vocab_size,
            embed_size=256,
            hidden_size=256,
            num_layers=2,
            dropout=0.1,
            pad_idx=pad_idx
        )
    else:
        model = PoetryTransformerModel(
            vocab_size=vocab_size,
            d_model=256,
            nhead=8,
            num_layers=2,
            dim_feedforward=512,
            dropout=0.1,
            pad_idx=pad_idx,
            max_len=64
        )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()
    # prompt 编码
    prompt = args.prompt.strip()
    if not prompt:
        print("请提供有效的诗歌开头 (--prompt)")
        return
    prompt_idx = [char2idx.get(ch, pad_idx) for ch in prompt]
    # 生成
    with torch.no_grad():
        if args.model == 'rnn':
            _, gen_str = model.generate(prompt_idx, args.length, char2idx, idx2char, temperature=args.temperature, device=device)
        else:
            gen_str = model.generate(prompt_idx, args.length, char2idx, idx2char, temperature=args.temperature, device=device)
    print(f"生成结果：\n{gen_str}")

if __name__ == '__main__':
    main()