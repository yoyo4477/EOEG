# VSCode完整使用教程 - 制造业数据分析项目

## 第一步：准备工作

### 1.1 确保您的文件结构

1. 打开Finder（访达）
2. 按快捷键 `Cmd + Shift + G`
3. 输入：`/Users/yoyo4477/Documents/数学建模/单子/001EOEG`
4. 确认文件夹结构如下：

```
001EOEG/
├── data/
│   └── Manufacturing_dataset.csv  (您的数据集在这里)
└── code/                          (将要放置所有代码的地方)
```

**如果 `code` 文件夹不存在**：
- 在 `001EOEG` 文件夹中右键 → 新建文件夹 → 命名为 `code`

**如果 `data` 文件夹不存在但数据集在其他地方**：
- 可以直接把 `Manufacturing_dataset.csv` 复制到 `code` 文件夹下

---

## 第二步：复制项目文件到code文件夹

### 2.1 下载项目文件

项目文件已经准备好，包含：
- `manufacturing_analysis/` 文件夹（包含所有代码）

### 2.2 复制到正确位置

1. 将整个 `manufacturing_analysis` 文件夹复制到：
   ```
   /Users/yoyo4477/Documents/数学建模/单子/001EOEG/code/
   ```

2. 复制完成后，结构应该是：
   ```
   /Users/yoyo4477/Documents/数学建模/单子/001EOEG/code/
   └── manufacturing_analysis/
       ├── config.py
       ├── main.py
       ├── requirements.txt
       ├── models/
       ├── utils/
       └── visualizations/
   ```

---

## 第三步：打开VSCode

### 3.1 启动VSCode

1. 打开 **VSCode** 应用程序
2. 如果是第一次使用，会看到欢迎页面

### 3.2 打开项目文件夹

**方法一（推荐）**：
1. 点击菜单栏：**文件 → 打开文件夹...** (或按 `Cmd + O`)
2. 在弹出的窗口中，按 `Cmd + Shift + G`
3. 粘贴路径：`/Users/yoyo4477/Documents/数学建模/单子/001EOEG/code/manufacturing_analysis`
4. 点击 **打开**

**方法二（使用终端）**：
1. 打开终端（Terminal）
2. 输入以下命令：
   ```bash
   cd /Users/yoyo4477/Documents/数学建模/单子/001EOEG/code/manufacturing_analysis
   code .
   ```

### 3.3 确认项目已打开

打开后，VSCode左侧的文件浏览器应该显示：
```
MANUFACTURING_ANALYSIS
├── config.py
├── main.py
├── requirements.txt
├── models
├── utils
└── visualizations
```

---

## 第四步：配置路径

### 4.1 打开配置文件

1. 在VSCode左侧文件列表中，**单击** `config.py`
2. 文件会在右侧编辑器中打开

### 4.2 检查并修改路径配置

找到以下部分（大约在第8-14行）：

```python
# ==================== 路径配置 ====================
# 基础路径：/Users/yoyo4477/Documents/数学建模/单子/001EOEG/code
BASE_DIR = "/Users/yoyo4477/Documents/数学建模/单子/001EOEG/code"

# 数据集路径（优先使用code目录下的数据集）
DATA_PATH = os.path.join(BASE_DIR, "Manufacturing_dataset.csv")
# 如果数据集在data文件夹，使用这个路径：
# DATA_PATH = "/Users/yoyo4477/Documents/数学建模/单子/001EOEG/data/Manufacturing_dataset.csv"
```

**根据您的数据集位置选择**：

**情况A：数据集在 `data` 文件夹下**
- 注释掉第12行，取消注释第14行：
```python
# DATA_PATH = os.path.join(BASE_DIR, "Manufacturing_dataset.csv")
DATA_PATH = "/Users/yoyo4477/Documents/数学建模/单子/001EOEG/data/Manufacturing_dataset.csv"
```

**情况B：数据集在 `code` 文件夹下**
- 保持默认即可（第12行）

### 4.3 保存文件

- 按 `Cmd + S` 保存文件
- 或点击菜单：**文件 → 保存**

---

## 第五步：打开终端

### 5.1 在VSCode中打开终端

**方法一**：
- 点击菜单：**终端 → 新建终端**

**方法二（快捷键）**：
- 按 `Ctrl + ~`（波浪号键，在Tab键上方）

**方法三**：
- 按 `Cmd + J` 打开面板，然后选择"终端"标签

### 5.2 确认终端位置

终端会在VSCode底部打开，显示类似：
```bash
yoyo4477@MacBook-Pro manufacturing_analysis %
```

确认当前目录是项目文件夹：
```bash
pwd
```

应该显示：
```
/Users/yoyo4477/Documents/数学建模/单子/001EOEG/code/manufacturing_analysis
```

---

## 第六步：安装依赖包

### 6.1 安装Python包

在终端中输入以下命令：

```bash
pip3 install -r requirements.txt
```

**如果速度慢**，使用清华镜像加速：
```bash
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 6.2 等待安装完成

- 安装过程可能需要 **2-5分钟**
- 看到 "Successfully installed ..." 表示成功
- 如果出现红色错误，请截图并检查

---

## 第七步：运行程序

### 7.1 运行主程序

在终端中输入：

```bash
python3 main.py
```

### 7.2 程序运行过程

程序会自动完成以下步骤：

```
STEP 1: 加载数据
  ↓
STEP 2: 构建9个模型
  ↓
STEP 3: 训练所有模型（每个epoch保存最佳模型）
  ↓
STEP 4: 评估性能
  ↓
STEP 5: 生成可视化
```

**预计时间**：
- 小数据集（<1000样本）：10-30分钟
- 中数据集（1000-10000样本）：30-60分钟
- 大数据集（>10000样本）：1-3小时

### 7.3 查看运行进度

终端会实时显示：
```
Training LSTM...
Epoch 1/100
...
Training GRU...
Epoch 1/100
...
```

---

## 第八步：查看结果

### 8.1 程序完成后

看到以下信息表示成功：
```
================================================================================
ANALYSIS COMPLETE!
================================================================================
```

### 8.2 打开结果文件夹

**方法一（使用Finder）**：
1. 打开Finder
2. 按 `Cmd + Shift + G`
3. 输入：`/Users/yoyo4477/Documents/数学建模/单子/001EOEG/code/output`
4. 回车

**方法二（使用终端）**：
```bash
open /Users/yoyo4477/Documents/数学建模/单子/001EOEG/code/output
```

### 8.3 查看输出文件

**表格文件** (在 `output/table/` 文件夹)：
- `performance_comparison.csv` - 性能对比表格
  - 用Excel或Numbers打开
  - 包含所有模型的MAE、MAPE、RMSE、Accuracy等指标

**图片文件** (在 `output/figure/` 文件夹)：
- `predictions_grid.png` - 预测结果对比（3行×5列）
- `probability_distribution.png` - 概率分布图
- `prediction_errors.png` - 预测误差图
- `roc_curves_grouped.png` - ROC曲线（1行×3列，含参考线）
- `roc_curves_all.png` - 所有模型ROC曲线
- `confusion_matrices.png` - 混淆矩阵
- `learning_curves.png` - 学习曲线（损失）
- `accuracy_curves.png` - 准确率曲线
- `metrics_comparison.png` - 指标对比柱状图
- `gradcam_heatmaps.png` - Grad-CAM热力图（3行×5列）

**模型文件** (在 `output/models/` 文件夹)：
- 每个模型有2个文件：
  - `模型名_best.h5` - 验证集上最佳版本
  - `模型名_final.h5` - 最终训练版本

---

## 第九步：快速测试（可选）

如果想快速测试程序是否正常（不完整训练）：

### 9.1 修改训练轮数

1. 在VSCode中打开 `config.py`
2. 找到第30行：
   ```python
   EPOCHS = 100  # 完整训练轮数
   ```
3. 改为：
   ```python
   EPOCHS = 5  # 快速测试
   ```
4. 保存文件（`Cmd + S`）
5. 运行 `python3 main.py`

---

## 常见问题解决

### 问题1：找不到数据文件

**错误信息**：
```
FileNotFoundError: Manufacturing_dataset.csv not found
```

**解决方法**：
1. 确认数据集文件存在
2. 检查 `config.py` 中的 `DATA_PATH` 设置
3. 确保路径中没有多余的文件夹

### 问题2：依赖包安装失败

**解决方法**：
```bash
# 升级pip
pip3 install --upgrade pip

# 使用清华镜像重试
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题3：内存不足

**错误信息**：
```
MemoryError or OOM
```

**解决方法**：
在 `config.py` 中修改：
```python
BATCH_SIZE = 16      # 原来是32
SEQUENCE_LENGTH = 5  # 原来是10
EPOCHS = 20          # 原来是100
```

### 问题4：训练太慢

**解决方法**：
- 减少 `EPOCHS`（如改为10）
- 或注释掉部分模型（编辑 `main.py` 中的 `build_all_models()` 函数）

---

## VSCode常用快捷键

| 功能 | Mac快捷键 |
|------|-----------|
| 保存文件 | `Cmd + S` |
| 打开文件 | `Cmd + P` |
| 打开文件夹 | `Cmd + O` |
| 新建终端 | `Ctrl + ~` |
| 查找 | `Cmd + F` |
| 替换 | `Cmd + H` |
| 注释/取消注释 | `Cmd + /` |
| 关闭当前文件 | `Cmd + W` |

---

## 项目文件说明

| 文件/文件夹 | 说明 |
|------------|------|
| `config.py` | 配置文件（路径、参数设置） |
| `main.py` | 主程序（运行这个文件） |
| `requirements.txt` | 依赖包列表 |
| `models/` | 所有模型定义 |
| `utils/` | 工具函数（数据加载、训练、评估） |
| `visualizations/` | 可视化工具 |

---

## 数据集要求

您的 `Manufacturing_dataset.csv` 必须包含以下7列：

1. **时间** (或 Time)
2. **温度** (或 Temperature)
3. **机器执行** (或 Machine_Performance)
4. **生产质量评分** (或 Production_Quality_Score)
5. **振动水平** (或 Vibration_Level)
6. **能源消耗量** (或 Energy_Consumption)
7. **最佳条件** (或 Optimal_Condition) - 标签（0或1）

**支持中英文列名**，程序会自动识别和转换。

---

## 下一步

完成训练后，您可以：

1. **查看性能表格**，确定最佳模型
2. **分析可视化结果**，理解模型行为
3. **使用保存的模型**进行新数据预测
4. **调整参数**（在 `config.py` 中），尝试提高性能
5. **修改模型**（在 `models/` 文件夹中）

---

**祝您使用顺利！** 🚀

如有问题，请检查终端中的错误信息。
