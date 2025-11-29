# Manufacturing Data Analysis - 深度学习模型对比

这是一个用于制造业数据分析的完整深度学习框架，实现了多个先进的深度学习模型，并提供全面的性能对比和可视化。

## 项目概述

本项目实现了以下功能：
1. **9个深度学习模型**：包括提议模型（论文中的三模块组合）和8个基准模型
2. **自动训练**：每个epoch自动保存最佳模型
3. **性能评估**：计算MAE、MAPE、RMSE、Accuracy、Precision、Recall、F1-Score、AUC等指标
4. **全面可视化**：
   - 预测结果可视化（3行×5列）
   - ROC曲线（1行×3列，含随机参考线）
   - Grad-CAM热力图（3行×5列）
   - 混淆矩阵、学习曲线、指标对比等

## 项目结构

```
manufacturing_analysis/
├── config.py                    # 配置文件
├── requirements.txt             # 依赖包
├── main.py                      # 主程序
├── models/                      # 模型定义
│   ├── proposed_model.py        # 提议模型（Ours）
│   ├── lstm_model.py            # LSTM模型
│   ├── cnn_lstm_model.py        # CNN-LSTM模型
│   ├── transformer_model.py     # Transformer模型
│   └── baseline_models.py       # 其他基准模型
├── utils/                       # 工具函数
│   ├── data_loader.py           # 数据加载
│   ├── trainer.py               # 模型训练
│   └── evaluator.py             # 性能评估
└── visualizations/              # 可视化工具
    ├── prediction_viz.py        # 预测结果可视化
    ├── roc_viz.py               # ROC曲线
    ├── gradcam_viz.py           # Grad-CAM热力图
    └── advanced_viz.py          # 高级可视化
```

## 详细使用教程（VSCode新手版）

### 步骤1：安装VSCode和Python

1. **安装Python**（如果还没有安装）
   - 访问 https://www.python.org/downloads/
   - 下载 Python 3.8 或更高版本
   - 安装时勾选 "Add Python to PATH"

2. **安装VSCode**
   - 访问 https://code.visualstudio.com/
   - 下载并安装

3. **安装Python扩展**
   - 打开VSCode
   - 点击左侧的扩展图标（或按 Cmd+Shift+X）
   - 搜索 "Python"
   - 安装 Microsoft 提供的 Python 扩展

### 步骤2：准备项目文件

1. **打开VSCode**

2. **打开项目文件夹**
   - 点击菜单：文件 → 打开文件夹（或按 Cmd+O）
   - 导航到：`/Users/yoyo4477/Documents/数学建模/单子/001EOEG/code`
   - 点击"打开"

3. **复制项目文件**
   - 将整个 `manufacturing_analysis` 文件夹复制到：
     `/Users/yoyo4477/Documents/数学建模/单子/001EOEG/code/`

### 步骤3：修改配置文件

1. **在VSCode左侧文件浏览器中**，点击打开：
   `manufacturing_analysis/config.py`

2. **确认路径设置正确**（默认已设置为您的路径）：
   ```python
   BASE_DIR = "/Users/yoyo4477/Documents/数学建模/单子/001EOEG/code"
   DATA_PATH = os.path.join(BASE_DIR, "Manufacturing_dataset.csv")
   ```

3. **保存文件**（Cmd+S）

### 步骤4：安装依赖包

1. **打开终端**
   - 在VSCode中，点击菜单：终端 → 新建终端
   - 或按快捷键：Ctrl+`

2. **进入项目目录**
   ```bash
   cd /Users/yoyo4477/Documents/数学建模/单子/001EOEG/code/manufacturing_analysis
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

   如果速度慢，可以使用国内镜像：
   ```bash
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

### 步骤5：运行程序

1. **运行主程序**
   - 在终端中输入：
   ```bash
   python main.py
   ```

2. **程序运行流程**
   - 加载数据
   - 构建9个模型
   - 训练所有模型（每个epoch保存最佳模型）
   - 评估所有模型
   - 生成所有可视化

3. **等待完成**
   - 训练可能需要较长时间（取决于数据量和epoch数）
   - 进度会实时显示在终端中

### 步骤6：查看结果

训练完成后，结果会保存在以下位置：

**1. 性能对比表格**
- 位置：`/Users/yoyo4477/Documents/数学建模/单子/001EOEG/code/saved_models/performance_comparison.csv`
- 用Excel或Numbers打开查看

**2. 保存的模型**
- 位置：`/Users/yoyo4477/Documents/数学建模/单子/001EOEG/code/saved_models/`
- 包含每个模型的最佳版本和最终版本

**3. 可视化图片**
- 位置：`/Users/yoyo4477/Documents/数学建模/单子/001EOEG/code/results/`
- 包含所有生成的图片：
  - `predictions_grid.png` - 预测结果（3×5）
  - `probability_distribution.png` - 概率分布
  - `prediction_errors.png` - 预测误差
  - `roc_curves_grouped.png` - ROC曲线分组（1×3）
  - `roc_curves_all.png` - 所有ROC曲线
  - `confusion_matrices.png` - 混淆矩阵
  - `learning_curves.png` - 学习曲线
  - `accuracy_curves.png` - 准确率曲线
  - `metrics_comparison.png` - 指标对比柱状图
  - `gradcam_heatmaps.png` - Grad-CAM热力图

## 自定义设置

### 修改训练参数

编辑 `config.py` 文件：

```python
# 训练参数
BATCH_SIZE = 32          # 批次大小
EPOCHS = 100             # 训练轮数（可以改小如20来快速测试）
LEARNING_RATE = 0.001    # 学习率

# 数据参数
SEQUENCE_LENGTH = 10     # 时间序列长度

# 可视化参数
FIGURE_DPI = 300         # 图片分辨率
FONT_SIZE = 16           # 字体大小
```

### 只训练特定模型

编辑 `main.py` 中的 `build_all_models()` 函数，注释掉不需要的模型。

## 模型说明

| 模型名称 | 说明 |
|---------|------|
| Proposed Model | 论文提出的完整模型（Baseline + Feature Engineering + Constraint-Aware） |
| LSTM | 长短期记忆网络 |
| GRU | 门控循环单元 |
| BiLSTM | 双向LSTM |
| CNN-LSTM | CNN与LSTM混合模型 |
| Attention-LSTM | 带注意力机制的LSTM |
| 1D-CNN | 一维卷积神经网络 |
| Transformer | Transformer模型 |
| MLP | 多层感知机 |

## 性能指标说明

- **MAE**: Mean Absolute Error（平均绝对误差）- 越小越好
- **MAPE**: Mean Absolute Percentage Error（平均绝对百分比误差）- 越小越好
- **RMSE**: Root Mean Squared Error（均方根误差）- 越小越好
- **Accuracy**: 准确率 - 越大越好
- **Precision**: 精确率 - 越大越好
- **Recall**: 召回率 - 越大越好
- **F1-Score**: F1分数 - 越大越好
- **AUC**: ROC曲线下面积 - 越大越好

## 常见问题

### 1. 内存不足
如果遇到内存错误，可以：
- 减小 `BATCH_SIZE`
- 减少 `SEQUENCE_LENGTH`
- 注释掉部分模型

### 2. 训练太慢
可以：
- 减少 `EPOCHS`（例如改为20）
- 使用更小的模型
- 减少数据量

### 3. 找不到数据文件
确保：
- `Manufacturing_dataset.csv` 在正确的位置
- `config.py` 中的路径设置正确

### 4. 图片中文显示问题
如果图片中文字显示异常，可以在 `config.py` 中修改字体设置。

## 联系方式

如有问题，请查看终端中的错误信息。

## License

MIT License
