import pandas as pd
import matplotlib.pyplot as plt
import json

# 读取 CSV 文件
csv_path = 'logs/ppl_log.csv'
df = pd.read_csv(csv_path)

# 检查是否包含所需列
assert 'epoch' in df.columns and 'rnn' in df.columns and 'transformer' in df.columns, "列名应为 epoch, rnn, transformer"

# 绘图保存 png
plt.figure(figsize=(8, 5))
plt.plot(df['epoch'], df['rnn'], marker='o', label='RNN')
plt.plot(df['epoch'], df['transformer'], marker='s', label='Transformer')
plt.xlabel('Epoch')
plt.ylabel('Perplexity (PPL)')
plt.title('RNN vs Transformer PPL Curve')
plt.legend()
plt.grid(True)
plt.savefig('ppl_comparison.png')  # ✅ 保存图片
plt.close()

# 保存 JSON 用于前端显示
ppl_json = {
    "epoch": df['epoch'].tolist(),
    "rnn": df['rnn'].tolist(),
    "transformer": df['transformer'].tolist()
}
with open('ppl_log.json', 'w', encoding='utf-8') as f:
    json.dump(ppl_json, f, ensure_ascii=False, indent=2)

print("✅ 生成完成：ppl_comparison.png 和 ppl_log.json")
