# 制造业数据分析 - VSCode使用说明

## 📁 第一步：准备文件夹结构

在Mac上创建以下结构：

```
/Users/yoyo4477/Documents/数学建模/单子/001EOEG/
├── data/
│   └── Manufacturing_dataset.csv  (您的数据集)
└── code/                          (将所有代码放这里)
```

---

## 📥 第二步：复制项目文件

**将以下文件和文件夹复制到 `code/` 文件夹：**

```
需要复制的文件：
✅ config.py
✅ main.py
✅ requirements.txt
✅ models/        (整个文件夹)
✅ utils/         (整个文件夹)
✅ visualizations/(整个文件夹)
```

**复制后的结构应该是：**

```
/Users/yoyo4477/Documents/数学建模/单子/001EOEG/code/
├── config.py
├── main.py
├── requirements.txt
├── models/
│   ├── __init__.py
│   ├── proposed_model.py
│   ├── lstm_model.py
│   ├── cnn_lstm_model.py
│   ├── transformer_model.py
│   └── baseline_models.py
├── utils/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── trainer.py
│   └── evaluator.py
└── visualizations/
    ├── __init__.py
    ├── prediction_viz.py
    ├── roc_viz.py
    ├── gradcam_viz.py
    └── advanced_viz.py
```

⚠️ **重要**：文件直接在 `code/` 下，**不要**有额外的子文件夹！

---

## 💻 第三步：在VSCode中打开项目

### 方法一：使用菜单

1. 打开 **VSCode**
2. 点击 **文件 → 打开文件夹...**
3. 按 `Cmd + Shift + G`
4. 粘贴路径：`/Users/yoyo4477/Documents/数学建模/单子/001EOEG/code`
5. 点击 **打开**

### 方法二：使用终端

```bash
cd /Users/yoyo4477/Documents/数学建模/单子/001EOEG/code
code .
```

### 确认打开正确

VSCode左侧应该显示：

```
CODE
├── config.py
├── main.py
├── requirements.txt
├── models
├── utils
└── visualizations
```

---

## ⚙️ 第四步：配置数据集路径

1. 在VSCode中打开 `config.py`
2. 找到第8-14行

**如果数据集在 `data/` 文件夹：**
```python
# 注释掉第12行，取消注释第14行
# DATA_PATH = os.path.join(BASE_DIR, "Manufacturing_dataset.csv")
DATA_PATH = "/Users/yoyo4477/Documents/数学建模/单子/001EOEG/data/Manufacturing_dataset.csv"
```

**如果数据集在 `code/` 文件夹：**
```python
# 保持默认（第12行）
DATA_PATH = os.path.join(BASE_DIR, "Manufacturing_dataset.csv")
```

3. 保存文件：`Cmd + S`

---

## 🔧 第五步：打开终端并安装依赖

### 5.1 在VSCode中打开终端

- 按 `Ctrl + ~`（波浪号）
- 或点击：**终端 → 新建终端**

### 5.2 安装Python包

```bash
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

等待2-5分钟安装完成。

---

## ▶️ 第六步：运行程序

在终端中输入：

```bash
python3 main.py
```

### 程序会自动完成：

```
✓ 加载数据（8:1:1划分）
✓ 构建9个模型
✓ 训练所有模型（每个epoch保存最佳模型）
✓ 评估性能
✓ 生成所有可视化
```

**预计时间**：30分钟 - 3小时（取决于数据量）

---

## 📊 第七步：查看结果

### 结果文件夹位置

```
/Users/yoyo4477/Documents/数学建模/单子/001EOEG/code/output/
├── table/         (性能对比表格)
│   └── performance_comparison.csv
├── figure/        (所有可视化图片)
│   ├── predictions_grid.png           (3行5列)
│   ├── roc_curves_grouped.png         (1行3列)
│   ├── gradcam_heatmaps.png          (3行5列)
│   ├── confusion_matrices.png
│   ├── learning_curves.png
│   └── ...
└── models/        (保存的模型)
    ├── Proposed Model_best.h5
    ├── LSTM_best.h5
    └── ...
```

### 打开结果文件夹

**方法一（Finder）：**
```
按 Cmd+Shift+G
输入：/Users/yoyo4477/Documents/数学建模/单子/001EOEG/code/output
```

**方法二（终端）：**
```bash
open output
```

---

## 🚀 快速测试（可选）

如果想快速测试（不完整训练）：

1. 打开 `config.py`
2. 修改第30行：
   ```python
   EPOCHS = 5  # 改为5进行快速测试
   ```
3. 保存并运行

---

## ❓ 常见问题

### Q1: 找不到数据文件

**检查**：
- 数据集文件是否存在
- `config.py` 中的路径是否正确
- 文件名是否为 `Manufacturing_dataset.csv`

### Q2: 内存不足

**修改 `config.py`：**
```python
BATCH_SIZE = 16      # 减小批次
EPOCHS = 20          # 减少轮数
```

### Q3: 训练太慢

**修改 `config.py`：**
```python
EPOCHS = 10  # 快速模式
```

---

## 📋 数据集格式要求

CSV文件必须包含以下7列：

| 列名（中文） | 列名（英文） | 说明 |
|------------|------------|------|
| 时间 | Time | 时间戳 |
| 温度 | Temperature | 温度值 |
| 机器执行 | Machine_Performance | 性能指标 |
| 生产质量评分 | Production_Quality_Score | 质量分数 |
| 振动水平 | Vibration_Level | 振动数据 |
| 能源消耗量 | Energy_Consumption | 能耗值 |
| 最佳条件 | Optimal_Condition | **标签** (0或1) |

支持中英文列名，程序会自动识别。

---

## 🎯 输出说明

### 表格（CSV）

- `performance_comparison.csv` - 包含所有模型的MAE、MAPE、RMSE、Accuracy、Precision、Recall、F1、AUC等指标

### 图片（PNG，300 DPI，Times New Roman 16号）

1. **predictions_grid.png** - 预测结果对比（3行5列）
2. **roc_curves_grouped.png** - ROC曲线（1行3列，含随机参考线）
3. **gradcam_heatmaps.png** - Grad-CAM热力图（3行5列）
4. **confusion_matrices.png** - 混淆矩阵
5. **learning_curves.png** - 损失学习曲线
6. **accuracy_curves.png** - 准确率曲线
7. **metrics_comparison.png** - 指标对比柱状图
8. **probability_distribution.png** - 概率分布
9. **prediction_errors.png** - 预测误差
10. **roc_curves_all.png** - 所有ROC曲线合并图

### 模型文件（H5）

每个模型2个文件：
- `模型名_best.h5` - 验证集最佳版本
- `模型名_final.h5` - 最终训练版本

---

**完成！** 🎉

有问题请查看终端错误信息。
