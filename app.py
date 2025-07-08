import torch
import gradio as gr
import os
import re
import random
from preprocess.dataset import load_tang_poems, build_vocab
from models.model_run import PoetryRNNModel
from models.model_transform import PoetryTransformerModel


# ============ 随机补齐汉字工具 ============

def get_hanzi_set(idx2char, exclude=None):
    exclude = exclude or []
    # 只用常用汉字，且不补 prompt 里已有的
    return [c for c in idx2char.values() if re.match(r'[\u4e00-\u9fa5]', c) and c not in exclude]


def format_poem_strict(text, line_len, total_len, idx2char):
    hanzi = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    # 随机补齐到 total_len
    hanzi_set = get_hanzi_set(idx2char, exclude=list(hanzi))
    while len(hanzi) < total_len:
        hanzi += random.choice(hanzi_set) if hanzi_set else '天'
    hanzi = hanzi[:total_len]
    # 分行、加标点
    lines = [hanzi[i:i + line_len] for i in range(0, total_len, line_len)]
    result = ""
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result += line + '，'
        else:
            result += line + '。'
        if (i + 1) % 2 == 0 and i != len(lines) - 1:
            result += '\n'
    return result.strip()


# ============ 加载模型/数据 ============

poems = load_tang_poems()
char2idx, idx2char = build_vocab(poems)
pad_idx = char2idx['<PAD>']
vocab_size = len(char2idx)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rnn_model = PoetryRNNModel(
    vocab_size=vocab_size,
    embed_size=256,
    hidden_size=256,
    num_layers=2,
    dropout=0.1,
    pad_idx=pad_idx
)
rnn_model.load_state_dict(torch.load('checkpoints/rnn_epoch15.pt', map_location=device))
rnn_model = rnn_model.to(device)
rnn_model.eval()

transformer_model = PoetryTransformerModel(
    vocab_size=vocab_size,
    d_model=256,
    nhead=8,
    num_layers=2,
    dim_feedforward=512,
    dropout=0.1,
    pad_idx=pad_idx,
    max_len=64
)
transformer_model.load_state_dict(torch.load('checkpoints/transformer_epoch15.pt', map_location=device))
transformer_model = transformer_model.to(device)
transformer_model.eval()

# 尝试加载 PPL 图像路径
ppl_plot_path = "ppl_comparison.png" if os.path.exists("ppl_comparison.png") else None


# ============ 生成函数 ============

def generate_poems(prompt, strategy, param, style):
    prompt = prompt.strip()
    if not prompt:
        return "请输入诗歌开头", "请输入诗歌开头", ppl_plot_path

    # 五言/七言绝句
    if style == '五言':
        line_len, total_len = 5, 20
    else:
        line_len, total_len = 7, 28

    prompt_idx = [char2idx.get(ch, pad_idx) for ch in prompt if ch in char2idx]
    gen_len = max(0, total_len - len(re.sub(r'[^\u4e00-\u9fa5]', '', prompt)))

    kwargs = {"temperature": 1.0, "top_k": 0, "top_p": 0.0, "device": device}
    if strategy == "温度采样":
        kwargs["temperature"] = float(param)
    elif strategy == "Top-K采样":
        kwargs["top_k"] = int(param)
    elif strategy == "Top-P采样":
        kwargs["top_p"] = float(param)

    def force_generate(model):
        # 补充生成
        text = ""
        if isinstance(model, PoetryRNNModel):
            text = model.generate(prompt_idx, gen_len, char2idx, idx2char, **kwargs)
        else:
            text = model.generate(prompt_idx, gen_len, char2idx, idx2char, **kwargs)
        if isinstance(text, tuple):
            text = text[1]
        return text

    # 拼接并补足
    rnn_new = force_generate(rnn_model)
    transformer_new = force_generate(transformer_model)

    rnn_result = format_poem_strict(prompt + rnn_new, line_len, total_len, idx2char)
    transformer_result = format_poem_strict(prompt + transformer_new, line_len, total_len, idx2char)
    return rnn_result, transformer_result, ppl_plot_path


# ============ Gradio 页面 ============

with gr.Blocks(css="""
    .gradio-container {max-width: 900px !important; margin: auto;}
    textarea, input, button, .gr-button {font-size: 16px !important;}
    .gradio-container .block.svelte-13n1gj8 {padding: 0.5em 0;}
    .gradio-container label {font-size: 16px !important;}
    .gradio-container .gr-input {height: 38px;}
    .gradio-container .gr-textbox textarea {height: 90px;}
    .gradio-container .gr-button {padding: 0.5em 1.2em;}
""") as iface:
    gr.Markdown(
        "<h2 style='text-align:center'>古诗生成模型对比系统（RNN vs Transformer）</h2>"
        "<p>输入一句古诗作为开头，选择采样策略、诗歌风格，系统将使用 RNN 和 Transformer 自动续写，并展示生成结果与困惑度（PPL）曲线图。</p>"
    )
    with gr.Row():
        prompt_inp = gr.Textbox(label="请输入诗歌开头（如：白日依山尽）", value="", lines=1)
        style_sel = gr.Radio(choices=["五言", "七言"], value="五言", label="诗歌风格", interactive=True)
    with gr.Row():
        strategy_sel = gr.Radio(["温度采样", "Top-K采样", "Top-P采样"], value="温度采样", label="采样策略",
                                interactive=True)
        param_num = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="温度", interactive=True)
        top_k_slider = gr.Slider(1, 20, value=5, step=1, label="Top-K", interactive=True, visible=False)
        top_p_slider = gr.Slider(0.01, 1.0, value=0.9, step=0.01, label="Top-P", interactive=True, visible=False)

    with gr.Row():
        rnn_out = gr.Textbox(label="RNN 模型生成结果", lines=4)
        transformer_out = gr.Textbox(label="Transformer 模型生成结果", lines=4)
    ppl_out = gr.Image(type="filepath", label="PPL 曲线对比图", visible=True)


    def on_strategy_change(strategy):
        # 更新策略变化时显示不同的滑动条
        if strategy == "温度采样":
            return gr.update(visible=True, label="温度"), gr.update(visible=False), gr.update(visible=False)
        elif strategy == "Top-K采样":
            return gr.update(visible=False), gr.update(visible=True, label="Top-K"), gr.update(visible=False)
        elif strategy == "Top-P采样":
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True, label="Top-P")


    strategy_sel.change(on_strategy_change, strategy_sel, [param_num, top_k_slider, top_p_slider])


    def call_generate(prompt, strategy, param, style):
        try:
            return generate_poems(prompt, strategy, param, style)
        except Exception as e:
            return f"错误：{e}", f"错误：{e}", None


    btn = gr.Button("生成诗句")
    btn.click(call_generate, [prompt_inp, strategy_sel, param_num, style_sel], [rnn_out, transformer_out, ppl_out])

iface.launch()
