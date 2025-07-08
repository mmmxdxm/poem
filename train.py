import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import csv

from preprocess.dataset import load_tang_poems, build_vocab, poems_to_indices, PoetryDataset
from models.model_run import PoetryRNNModel
from models.model_transform import PoetryTransformerModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train RNN and Transformer for Poetry Generation")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--embed_size', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--ppl_log', type=str, default='logs/ppl_log.csv')
    parser.add_argument('--generate_len', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=1.0)
    return parser.parse_args()


def calculate_ppl(loss):
    try:
        return math.exp(loss)
    except OverflowError:
        return float('inf')


def train_one(model, dataloader, criterion, optimizer, args, pad_idx):
    model.train()
    total_loss = 0
    for input_seq, target_seq in dataloader:
        input_seq = input_seq.to(args.device)
        target_seq = target_seq.to(args.device)
        optimizer.zero_grad()
        if isinstance(model, PoetryRNNModel):
            logits, _ = model(input_seq)
        else:
            logits = model(input_seq)
        loss = criterion(logits.view(-1, logits.size(-1)), target_seq.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss, calculate_ppl(avg_loss)


def generate_example(model, prompt_idx, idx2char, char2idx, args, pad_idx):
    if isinstance(model, PoetryRNNModel):
        gen_str = model.generate(prompt_idx, args.generate_len, char2idx, idx2char, temperature=args.temperature, device=args.device)
    else:
        gen_str = model.generate(prompt_idx, args.generate_len, char2idx, idx2char, temperature=args.temperature, device=args.device)
    return gen_str


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.ppl_log), exist_ok=True)
    print(f"Using device: {args.device}")

    poems = load_tang_poems()
    char2idx, idx2char = build_vocab(poems)
    pad_idx = char2idx['<PAD>']
    vocab_size = len(char2idx)
    poems_indices = poems_to_indices(poems, char2idx)
    dataset = PoetryDataset(poems_indices, seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Init models
    rnn = PoetryRNNModel(vocab_size, args.embed_size, args.hidden_size, args.num_layers, args.dropout, pad_idx).to(args.device)
    transformer = PoetryTransformerModel(vocab_size, args.embed_size, 8, args.num_layers, args.hidden_size*2, args.dropout, pad_idx, args.seq_len).to(args.device)

    rnn_optimizer = torch.optim.Adam(rnn.parameters(), lr=args.lr)
    transformer_optimizer = torch.optim.Adam(transformer.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)

    # Initialize PPL CSV
    with open(args.ppl_log, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'rnn', 'transformer'])

    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")

        rnn_loss, rnn_ppl = train_one(rnn, dataloader, criterion, rnn_optimizer, args, pad_idx)
        transformer_loss, transformer_ppl = train_one(transformer, dataloader, criterion, transformer_optimizer, args, pad_idx)

        print(f"[RNN] Loss: {rnn_loss:.4f}, PPL: {rnn_ppl:.2f}")
        print(f"[Transformer] Loss: {transformer_loss:.4f}, PPL: {transformer_ppl:.2f}")

        # Save PPL log
        with open(args.ppl_log, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, rnn_ppl, transformer_ppl])

        # Save models
        torch.save(rnn.state_dict(), os.path.join(args.save_dir, f"rnn_epoch{epoch}.pt"))
        torch.save(transformer.state_dict(), os.path.join(args.save_dir, f"transformer_epoch{epoch}.pt"))

        # Generate samples
        prompt = poems[0][:7]
        prompt_idx = [char2idx.get(ch, pad_idx) for ch in prompt]
        rnn_sample = generate_example(rnn, prompt_idx, idx2char, char2idx, args, pad_idx)
        transformer_sample = generate_example(transformer, prompt_idx, idx2char, char2idx, args, pad_idx)

        print(f"[RNN Sample] {rnn_sample}")
        print(f"[Transformer Sample] {transformer_sample}")


if __name__ == '__main__':
    main()
