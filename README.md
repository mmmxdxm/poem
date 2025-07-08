唐诗生成器 - Poetry Generator
本项目是一个基于 PyTorch 的中文古诗生成系统，支持 RNN（LSTM）和 Transformer 两种模型结构，并提供简洁易用的 Gradio Web 页面进行交互式诗歌生成，旨在完成项目一的全部要求。

🎯 项目特性

✅ 支持两种模型结构：RNN（LSTM）和 Transformer

✅ 自动加载唐诗数据集并构建词表

✅ 提供训练脚本 train.py 支持命令行参数配置

✅ 提供 app.py 实现图形界面生成诗句

✅ 数据与模型模块分离，结构清晰易扩展

📁 项目结构说明

poetry_generator/

├── data/

│   └── poet.tang.0.json        # 唐诗 JSON 数据集

├── preprocess/

│   └── dataset.py              # 数据预处理、构建 Dataset 类

├── models/

│   ├── model_run.py            # 基于 LSTM 的诗歌生成模型

│   └── model_transform.py      # 基于 Transformer 的诗歌生成模型

├── train.py                    # 主训练脚本

├── generate.py                 # 单次生成脚本（可选）

├── generate——ppl_plot.py       # 生成ppl对比图 

├── app.py                      # Gradio 前端界面入口

├── requirements.txt            # 所有依赖包

└── README.md                   # 项目说明文档（当前文件）

📦 环境依赖安装
请先创建虚拟环境（如 Anaconda）并激活，然后执行：


pip install -r requirements.txt
如果提示 gradio 找不到，也可以单独安装：

pip install gradio

🧠 训练模型
使用命令行运行 train.py 开始训练：


# 使用 transformer 模型训练
python train.py --model transformer --epochs 10

# 或使用 rnn 模型训练
python train.py --model rnn --epochs 10
更多可选参数如 --seq_len、--batch_size、--generate_len 可通过 --help 查看。

🧪 示例生成

训练完成后自动生成一段示例诗句，并保存在控制台输出：


[Sample] 春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。

🌐 启动前端界面

运行如下命令即可打开 Gradio 页面：


python app.py

页面将自动打开浏览器，用户可输入起始词，选择模型，点击生成诗歌。

📊 模型对比效果

模型类型	起始文本	生成片段

RNN（LSTM）	山中夜雨	山中夜雨急，客舍梦初成。

Transformer	山中夜雨	山中风雨夜，独坐听虫鸣。

🧩 拓展方向建议

加入 GPT 小模型实现

自动评估生成内容的平仄、对仗、押韵

使用 Flask / FastAPI 构建后端接口

📚 数据来源

唐诗数据来自开源项目：https://github.com/chinese-poetry/chinese-poetry