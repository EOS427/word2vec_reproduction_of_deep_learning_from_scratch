# 鱼书word2vec复现

> 本文档由 **Trae** 撰写，内容依据 2026-11-11 项目的最新代码与目录状态整理。

基于《深度学习进阶：自然语言处理》（俗称"鱼书"）第 3~4 章的 Word2Vec 复现，使用纯 NumPy / CuPy 手写前向与反向传播，不依赖 PyTorch、TensorFlow。

项目在教材原有实现上做了三处延伸：

- 数据规模从玩具语料 `You say goodbye and I say hello.` 扩展到 **PTB 数据集**（约 94.6 万词，词表 9945）；
- 训练目标从 **朴素 Softmax** 升级为 **负采样（Negative Sampling）**，使全量训练成为可能；
- 增加 **配置文件 + 缓存 + 交互式评估** 的工程化封装。

---

## 目录

- [项目特性](#项目特性)
- [目录结构](#目录结构)
- [环境要求](#环境要求)
- [安装](#安装)
- [快速开始](#快速开始)
- [配置文件说明](#配置文件说明)
- [数据与缓存](#数据与缓存)
- [实现说明](#实现说明)
- [评估方法](#评估方法)
- [已知问题](#已知问题)
- [变更记录](#变更记录)
- [参考](#参考)

---

## 项目特性

| 特性 | 说明 |
|---|---|
| 两套模型 | `regular_model`（Softmax 版，教材原味）/ `upgrade_model`（负采样版，可跑 PTB） |
| 两种架构 | CBOW 与 Skip-gram 均已实现 |
| 纯手写反向传播 | `MatMul` / `Affine` / `SoftmaxWithLoss` / `Embedding` / `Negative_Sampling_Loss` |
| 多种优化器 | SGD、Momentum、Nesterov、AdaGrad、RMSprop、Adam（默认 Adam） |
| 后端可切换 | `numpy`（CPU）或 `cupy`（GPU），由配置文件的 `GPU` 字段统一分发 |
| 结果可评估 | `most_similar`（近义词）与 `analogy`（a:b == c:?）两种方式 |
| 缓存机制 | 语料 ID 序列、词表、词向量三元组均落盘缓存，避免重复计算 |

---

## 目录结构

```text
word2vec_package/
├── src/
│   ├── common/
│   │   ├── choose_proc.py         # 计算后端选择：cupy(GPU) / numpy(CPU)
│   │   ├── load_config.py         # 读取 pyproject.toml 中的 [tool.*] 配置
│   │   ├── layers.py              # 基础层：MatMul、Affine、Softmax、Sigmoid、SoftmaxWithLoss
│   │   ├── accelerate_layers.py   # 加速层：Embedding、Embedding_Dot、负采样、Unigram_Sampler
│   │   ├── functions.py           # softmax、sigmoid、cross_entropy_error
│   │   ├── context_preprocess.py  # 建词表、生成 context/target、one-hot 化
│   │   ├── trainer.py             # 训练循环、损失打印、共享参数去重
│   │   ├── optimizer.py           # 6 种优化器
│   │   └── util.py                # 余弦相似度、most_similar、analogy
│   ├── data/
│   │   └── ptb/
│   │       ├── load_ptb.py        # PTB 下载 / 词表构建 / npy 缓存
│   │       ├── ptb.train.txt      # 训练语料（原始文本，纯 ASCII）
│   │       ├── ptb.train.npy      # 语料 ID 序列缓存
│   │       └── ptb.vocab.pkl      # 词表缓存
│   └── model/
│       ├── regular_model/         # 朴素 Softmax 版 CBOW / Skip-gram
│       └── upgrade_model/         # 负采样版 CBOW / Skip-gram
├── script/
│   └── evaluate.py                # 主入口：加载/训练词向量 + 交互式评估
├── test/
│   ├── regular_model_train.py     # 小语料训练示例（验证反向传播是否正确）
│   └── upgrade_model_train.py     # PTB 训练示例（打印全部词向量 + 绘制 loss）
├── cache/                         # 词向量缓存目录（当前为空，首次运行后生成 word_matrix.pkl）
├── pyproject.toml                 # 依赖 + 项目自身配置
├── requirement.txt
├── uv.lock
└── README.md
```

> 说明：
> - `main.py` 为空文件，实际入口是 `script/evaluate.py`；
> - `src/model/regular_model/` 与 `src/model/upgrade_model/` 下各有一个 0 字节的说明文件占位（`面向数据one_hot化且窗口大小为一的简单情况.txt`、`计算加速且可处理变窗口大小情况.txt`），用于后续补充笔记，不参与运行。

---

## 环境要求

- Python **3.14**（见 `pyproject.toml` 的 `requires-python = ">=3.14"`）
- NumPy 2.5（CPU 模式必需，已在依赖中）
- CuPy + NVIDIA CUDA（GPU 模式可选，**未**写入依赖）
- Matplotlib（`test/` 下脚本绘制 loss 曲线时需要）

依赖清单见 [requirement.txt](requirement.txt) 与 [pyproject.toml](pyproject.toml)。

> 当前仓库未附带 `.venv`，请按下面的安装步骤自行创建环境。

---

## 安装

推荐使用 `uv`（仓库已提供 `uv.lock`，可复现依赖版本）：

```powershell
uv sync                 # 按 uv.lock 创建环境并安装依赖
uv run python script\evaluate.py
```

也可以使用标准 venv + pip：

```powershell
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirement.txt
```

如需 GPU 加速，额外安装对应 CUDA 版本的 CuPy：

```powershell
# CUDA 12.x
uv pip install cupy-cuda12x
```

---

## 快速开始

### 1. 小语料自检

用教材中的玩具句子验证网络结构与反向传播是否正确：

```powershell
uv run python test\regular_model_train.py
# 提示输入模型后键入：cbow 或 skip_gram
```

### 2. 交互式评估（主入口）

```powershell
uv run python script\evaluate.py
```

运行逻辑：

1. 若 `cache/word_matrix.pkl` 存在 → 直接读出 `word_to_id`、`id_to_word`、`word_matrix` 三元组；
2. 否则提示选择模型（`cbow` / `skip_gram`），用 PTB 训练后把三元组写入缓存；
3. 进入交互菜单：

```text
选择测试方法：
most_similar
analogy
exit
```

示例：

```text
输入名字: most_similar
请输入查询词：the

[query] the
 manages: 0.85
 pools: 0.85
 passed: 0.83
 done
```

```text
输入名字: analogy
a:b==c:?
a:king
b:man
c:woman

[analogy] king:man = woman:?
 preceding: 4.76
 settled: 4.74
 done
```

> 以上为示意输出，具体数值以实际训练结果为准。由于训练数据是 PTB（新闻文本），`king/man/woman` 类比效果不会像教材示例那样理想，属于预期现象。

### 3. 全量训练并观察 loss

```powershell
uv run python test\upgrade_model_train.py
```

训练结束后用 Matplotlib 弹出 loss 曲线，并打印每个词及其 100 维向量。

> `test/upgrade_model_train.py` 使用硬编码超参数，不读取 `pyproject.toml`。

---

## 配置文件说明

超参数集中在 `pyproject.toml` 的 `[tool.*]` 下，由 [load_config.py](src/common/load_config.py) 读取：

```toml
[tool.word2vec_package]
win_size   = 5      # 上下文窗口半径，上下文总长为 2*win_size
hid_size   = 100    # 词向量维度
batch_size = 100
max_epoch  = 10

[tool.processor]
GPU = true          # true: cupy 后端；false: numpy 后端
```

`load_tool(name)` 支持两个取值：

| name | 返回内容 | 使用方 |
|---|---|---|
| `word2vec_package` | 训练超参数 | `script/evaluate.py` |
| `processor` | 后端开关字典 | `src/common/choose_proc.py` |

后端判定方式（[choose_proc.py](src/common/choose_proc.py)）：

```python
is_use_GPU = load_tool("processor")
GPU = is_use_GPU["GPU"]     # 直接按键取值
```

因此：

- `GPU = false` 现在可以真正切到 NumPy 后端；
- 但**不能删除整个 `[tool.processor]` 节**，否则会抛 `KeyError: 'GPU'`。

---

## 数据与缓存

### PTB 数据集

- 来源：`https://raw.githubusercontent.com/tomsercu/lstm/master/data/`
- 首次运行自动下载 `ptb.train.txt`（约 5.1 MB，纯 ASCII 编码）
- 处理流程：换行 → `<eos>` → 按空格分词 → 建词表 → 转 ID 序列

实测规模（`win_size = 5`）：

| 指标 | 数值 |
|---|---|
| 训练语料 token 数 | 946,312 |
| 词表大小 | 9,945 |
| 上下文矩阵形状 | 946,302 × 10 |
| 目标向量形状 | 946,302 |

### 缓存文件

| 文件 | 格式 | 内容 |
|---|---|---|
| `src/data/ptb/ptb.vocab.pkl` | pickle | `(corpus, word_to_id, id_to_word)` |
| `src/data/ptb/ptb.train.npy` | NumPy npy | 语料 ID 序列 `int64[946312]` |
| `cache/word_matrix.pkl` | pickle | `{"word_to_id": …, "id_to_word": …, "word_matrix": …}` |

要点：

- 词向量缓存已从旧的 `word_matrix.txt` 改为 **`word_matrix.pkl`**，并用字典同时保存词表与矩阵，解决了旧版"缓存命中后词表为 None"的问题；
- 旧格式 `cache/word_matrix.txt`（单独一个 CuPy 数组）**不再被读取**，如仍存在可直接删除；
- `cache/` 目录当前为空，因此首次运行 `evaluate.py` 会进入训练分支。

---

## 实现说明

### 模型对比

| | `regular_model` | `upgrade_model` |
|---|---|---|
| 输出层 | 全词表 Softmax | 负采样（1 正例 + 5 负例） |
| 复杂度 | O(V)，V=9945 时不可行 | O(K)，K=6，可训练 PTB |
| 词向量矩阵 | `W_in: (V, H)` | `W_in: (V, H)`、`W_out: (V, H)` |
| 输入形式 | one-hot（`regular_processing`） | 词 ID（`load_data`） |
| 适用场景 | 玩具语料，验证梯度 | 真实语料，实际训练 |

### CBOW vs Skip-gram

- **CBOW**：上下文词向量求均值后预测中心词。`upgrade_model/cbow.py` 中 `context_len = 2 * win_size`，对 10 个 Embedding 结果累加后除以 `context_len`。
- **Skip-gram**：用中心词预测每个上下文词，损失为 `2 * win_size` 个负采样损失之和。

### 关键实现细节

1. **共享权重去重**：`W_in` 被多个 `MatMul` / `Embedding` 引用，`trainer.remove_duplicate()` 在更新前合并重复参数及梯度（含转置共享的判定）。
2. **负采样采样器**：`Unigram_Sampler` 按词频的 0.75 次幂构造分布；采样时把正例概率清零并重新归一化，然后不放回抽取（`replace=False`）。
3. **数值稳定**：`softmax` 先减最大值；交叉熵加 `1e-7`；`normalize` 与余弦相似度加 `1e-8` 防除零。
4. **后端无关**：所有张量操作统一走 `choose_proc.proc`，切换后端只改配置的一行。

---

## 评估方法

### most_similar

对查询词与词表中每个词计算余弦相似度，排序后输出 Top-N（默认 5）。

### analogy

计算 `vec(b) - vec(a) + vec(c)` 并归一化，与全部词向量做点积（等价余弦相似度），输出 Top-N，自动排除 `a`、`b`、`c` 以及 NaN 结果。

函数签名（[util.py](src/common/util.py)）：

```python
most_similar(query, word_to_id, id_to_word, word_matrix, top=5)
analogy(a, b, c, word_to_id, id_to_word, word_matrix, top=5, answer=None)
```

---

## 已知问题

以下问题均已对照当前代码核实：

| # | 现象 | 原因 | 解决思路 |
|---|---|---|---|
| 1 | `ModuleNotFoundError: No module named 'cupy'` | `GPU = true`，而环境与 `uv.lock` 均未包含 CuPy，`choose_proc.py` 在导入期就 `import cupy` | 安装 `cupy-cuda12x`；或把 `GPU` 改为 `false` 使用 NumPy |
| 2 | 缓存读写抛 `ValueError: binary mode doesn't take an encoding argument` | `evaluate.py` 中 `open(matrix_path, "rb", encoding="utf-8")` 与 `open(matrix_path, "wb", encoding="utf-8")` 同时指定了二进制模式与编码 | 去掉 `encoding="utf-8"`（pickle 二进制读写不需要编码）；注意写缓存发生在训练**结束之后**，会白跑一轮训练才报错 |
| 3 | CPU 训练抛 `AttributeError: 'numpy.ndarray' object has no attribute 'get'` | 负采样中 `p=prep_p.get()`，`.get()` 是 CuPy 专有方法 | 按后端区分：CuPy 用 `.get()`，NumPy 直接用 `numpy.asarray(prep_p)` |
| 4 | 删除 `[tool.processor]` 后报 `KeyError: 'GPU'` | `GPU = is_use_GPU["GPU"]` 直接按键取值 | 保留该节并写 `GPU = false`，或改为 `is_use_GPU.get("GPU", False)` |
| 5 | 语料读取依赖系统默认编码 | `load_ptb.py` 中 `open(file_path)` 未指定 `encoding` | 显式写 `encoding="utf-8"`（PTB 为纯 ASCII，当前不会报错，但换语料会踩坑） |
| 6 | `evaluate.py` 交互中直接回车或输入非菜单项 | 输入未做空值处理，会反复走 `error:retry` | 属交互体验问题，可加空输入判断 |

现状总结：

- 默认配置（`GPU = true`）下，`evaluate.py` 会在**导入阶段**因缺少 CuPy 直接退出；
- 改成 `GPU = false` 后可以导入，但训练会卡在问题 3；
- 缓存分支目前无缓存文件，暂时走不到问题 2；一旦生成缓存，读写都会命中该错误。

---

## 变更记录

相对上一版 README 所描述的状态，项目当前有以下改动：

| 项 | 旧状态 | 当前状态 |
|---|---|---|
| 配置读取 | `pyproject.txt`（文件不存在） | 改为读取 `pyproject.toml`，可正常加载 |
| 后端开关 | `GPU = is_use_GPU`（字典恒为真） | 改为 `GPU = is_use_GPU["GPU"]`，`false` 生效 |
| 缓存文件 | `cache/word_matrix.txt`（CuPy 数组） | `cache/word_matrix.pkl`，字典三元组 |
| 缓存分支 | 只恢复矩阵，查询报 `UnboundLocalError` | 同时恢复词表与矩阵，问题已修复 |
| 虚拟环境 | 仓库内含 `.venv` | 已移除，改用 `uv sync` 或自建环境 |
| 目录 | 无 `cache` 产物 | `cache/` 为空目录，首次运行生成缓存 |
| 新增文件 | — | `README.md`、`.screenshots/`、两个模型说明占位 txt |
| 遗留问题 | — | 问题 2（二进制模式带 encoding）为本次改动新引入 |

---

## 参考

- 斋藤康毅.《深度学习进阶：自然语言处理》. 人民邮电出版社（第 3 章 word2vec、第 4 章 word2vec 的高速化）
- Mikolov et al. *Distributed Representations of Words and Phrases and their Compositionality*. NIPS 2013.
- Mikolov et al. *Efficient Estimation of Word Representations in Vector Space*. ICLR 2014.
- PTB 数据镜像：<https://github.com/tomsercu/lstm/tree/master/data>
