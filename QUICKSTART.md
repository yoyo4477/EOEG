# 快速入门指南 - VSCode完全新手版

## 前置条件

确保您的数据集文件在以下位置：
```
/Users/yoyo4477/Documents/数学建模/单子/001EOEG/code/Manufacturing_dataset.csv
```

## 一、安装准备（首次使用）

### 1. 安装Python（如果还没有）

```bash
# 在终端中检查Python是否已安装
python3 --version
```

如果没有安装，访问 https://www.python.org/downloads/ 下载安装。

### 2. 安装VSCode（如果还没有）

访问 https://code.visualstudio.com/ 下载并安装。

### 3. 在VSCode中安装Python扩展

1. 打开VSCode
2. 按 `Cmd+Shift+X` 打开扩展面板
3. 搜索 "Python"
4. 安装Microsoft提供的Python扩展

## 二、项目设置

### 1. 将项目文件复制到正确位置

将整个 `manufacturing_analysis` 文件夹复制到：
```
/Users/yoyo4477/Documents/数学建模/单子/001EOEG/code/
```

最终结构应该是：
```
/Users/yoyo4477/Documents/数学建模/单子/001EOEG/code/
├── Manufacturing_dataset.csv
└── manufacturing_analysis/
    ├── config.py
    ├── main.py
    ├── requirements.txt
    ├── models/
    ├── utils/
    └── visualizations/
```

### 2. 在VSCode中打开项目

**方法一：使用VSCode菜单**
1. 打开VSCode
2. 点击：文件 → 打开文件夹
3. 选择：`/Users/yoyo4477/Documents/数学建模/单子/001EOEG/code/manufacturing_analysis`
4. 点击"打开"

**方法二：使用终端**
```bash
cd /Users/yoyo4477/Documents/数学建模/单子/001EOEG/code/manufacturing_analysis
code .
```

### 3. 打开终端

在VSCode中：
- 点击菜单：终端 → 新建终端
- 或按快捷键：`Ctrl + ~`（波浪号）

### 4. 安装依赖包

在终端中输入以下命令：

```bash
# 使用国内镜像加速（推荐）
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或使用默认源
pip3 install -r requirements.txt
```

等待安装完成（可能需要几分钟）。

## 三、运行程序

### 1. 确认路径设置

在VSCode中打开 `config.py` 文件，确认以下路径正确：

```python
BASE_DIR = "/Users/yoyo4477/Documents/数学建模/单子/001EOEG/code"
DATA_PATH = os.path.join(BASE_DIR, "Manufacturing_dataset.csv")
```

### 2. 运行主程序

**方法一：在终端中运行**
```bash
python3 main.py
```

**方法二：在VSCode中点击运行**
1. 打开 `main.py` 文件
2. 点击右上角的绿色三角形按钮（运行）
3. 或按 `Cmd+Shift+D` 打开调试面板，点击运行

### 3. 等待完成

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
- 小型数据集（<1000样本）：约10-30分钟
- 中型数据集（1000-10000样本）：约30-60分钟
- 大型数据集（>10000样本）：约1-3小时

**提示**：如果想快速测试，可以在 `config.py` 中将 `EPOCHS = 100` 改为 `EPOCHS = 10`。

## 四、查看结果

### 1. 性能对比表格

在Finder中打开：
```
/Users/yoyo4477/Documents/数学建模/单子/001EOEG/code/results/performance_comparison.csv
```

用Excel或Numbers打开，可以看到类似论文Table II的性能对比表格。

### 2. 可视化图片

在Finder中打开文件夹：
```
/Users/yoyo4477/Documents/数学建模/单子/001EOEG/code/results/
```

您会看到以下图片：

| 文件名 | 说明 |
|--------|------|
| `predictions_grid.png` | 预测结果对比（3行×5列） |
| `probability_distribution.png` | 概率分布图 |
| `prediction_errors.png` | 预测误差图 |
| `roc_curves_grouped.png` | ROC曲线分组对比（1行×3列） |
| `roc_curves_all.png` | 所有模型ROC曲线 |
| `confusion_matrices.png` | 混淆矩阵 |
| `learning_curves.png` | 学习曲线（损失） |
| `accuracy_curves.png` | 准确率曲线 |
| `metrics_comparison.png` | 指标对比柱状图 |
| `gradcam_heatmaps.png` | Grad-CAM热力图 |

### 3. 保存的模型

在Finder中打开文件夹：
```
/Users/yoyo4477/Documents/数学建模/单子/001EOEG/code/saved_models/
```

每个模型都有两个版本：
- `模型名_best.h5` - 训练过程中验证集上表现最好的版本
- `模型名_final.h5` - 训练完成后的最终版本

## 五、快速测试（可选）

如果您想快速测试系统是否正常工作，可以：

### 1. 创建测试脚本

在VSCode中创建新文件 `quick_test.py`：

```python
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import DataLoader

print("Testing data loader...")
loader = DataLoader()
X_train, X_val, X_test, y_train, y_val, y_test, scaler = loader.prepare_data()
print("✓ Data loaded successfully!")
print(f"  Train: {X_train.shape}")
print(f"  Val: {X_val.shape}")
print(f"  Test: {X_test.shape}")
```

### 2. 运行测试

```bash
python3 quick_test.py
```

如果看到输出类似：
```
✓ Data loaded successfully!
  Train: (xxx, 10, 5)
  Val: (xxx, 10, 5)
  Test: (xxx, 10, 5)
```

说明系统配置正确！

## 六、常见问题解决

### 问题1：找不到数据文件

**错误信息**：
```
FileNotFoundError: Manufacturing_dataset.csv not found
```

**解决方法**：
1. 确认数据文件在：`/Users/yoyo4477/Documents/数学建模/单子/001EOEG/code/Manufacturing_dataset.csv`
2. 检查 `config.py` 中的路径设置

### 问题2：依赖包安装失败

**解决方法**：
```bash
# 升级pip
pip3 install --upgrade pip

# 使用国内镜像重试
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题3：内存不足

**错误信息**：
```
MemoryError or OOM (Out of Memory)
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
在 `config.py` 中修改：
```python
EPOCHS = 10  # 快速测试用
```

或在 `main.py` 中注释掉部分模型：
```python
# 只训练3个模型进行测试
def build_all_models(input_shape):
    models = {}
    models['Proposed Model'] = build_proposed_model(input_shape)
    models['LSTM'] = build_lstm_model(input_shape)
    models['Transformer'] = build_transformer_model(input_shape)
    # 其他模型暂时注释掉
    return models
```

## 七、下一步

完成训练后，您可以：

1. **查看性能表格**，确定最佳模型
2. **分析可视化结果**，理解模型行为
3. **使用保存的模型**进行新数据预测
4. **调整参数**，尝试提高性能

## 需要帮助？

如果遇到问题：
1. 查看终端中的完整错误信息
2. 检查README.md中的常见问题部分
3. 确保所有文件路径正确
4. 确保数据集格式正确（7列：时间、温度、机器执行、生产质量评分、振动水平、能源消耗量、最佳条件）

---

**祝您使用顺利！** 🚀
